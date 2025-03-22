import os
import time
import re
import json
import sqlite3
import io
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder


# Set up the page layout and configuration
st.set_page_config(
    page_title="Subtitle Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Helper functions
def load_configuration(file_path="config.json"):
    """Load settings from a JSON file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "vector_db_path": "chroma_vectors",
                "subtitle_db": "processed_data.db",
                "google_api_key": "",
                "search_timeout_seconds": 20,
                "max_display_results": 20,
                "default_results_count": 5
            }
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return None


def setup_google_api(config):
    """Setup Google API key."""
    if "google_api_key" in config and config["google_api_key"]:
        genai.configure(api_key=config["google_api_key"])
    else:
        api_key_input = st.sidebar.text_input("Enter Google API Key", type="password")
        if api_key_input:
            genai.configure(api_key=api_key_input)
            config["google_api_key"] = api_key_input


def clean_text(text):
    """Remove unwanted characters and timestamps from text."""
    if not text:
        return ""
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()


@st.cache_resource
def get_embeddings_model():
    """Load the model for embeddings."""
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    except Exception as e:
        st.error(f"Error loading embeddings model: {e}")
        return None


def transcribe_audio_clip(audio_file):
    """Convert audio to text using the Gemini 2.0 Flash API."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.getvalue())
            audio_path = temp_audio.name

        uploaded_audio = genai.upload_file(audio_path, mime_type="audio/wav")
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        with st.status("Transcribing..."):
            response = model.generate_content(
                [uploaded_audio, "Transcribe this audio to text"],
                generation_config={"max_output_tokens": 2000}
            )

        os.unlink(audio_path)
        return clean_text(response.text)
    except Exception as e:
        st.error(f"Audio transcription error: {e}")
        return None


def execute_search(query, db_path, top_k=5, timeout=20):
    """Search the vector database for relevant subtitles."""
    try:
        embedding_model = get_embeddings_model()
        if embedding_model is None:
            return []

        vector_store = Chroma(persist_directory=db_path, embedding_function=embedding_model)

        with st.status("Performing search..."):
            start_time = time.time()
            results = vector_store.similarity_search_with_score(clean_text(query), k=top_k)
            search_duration = time.time() - start_time

            if search_duration > timeout:
                st.warning(f"Search took {search_duration:.1f}s, exceeding the timeout of {timeout}s.")
            else:
                st.success(f"Search completed in {search_duration:.1f}s.")

        # Format results
        formatted_results = []
        for document, score in results:
            doc_id = document.metadata.get("num", hash(document.page_content))
            similarity = 1.0 - score if score <= 1.0 else 0.0
            formatted_results.append((doc_id, similarity))

        return formatted_results
    except Exception as e:
        st.error(f"Error during semantic search: {e}")
        return []


def retrieve_subtitle_text(doc_id, db_path):
    """Fetch subtitle content from the database using document ID."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, cleaned_text FROM processed_subtitles WHERE num = ?", (doc_id,))
        result = cursor.fetchone()
        conn.close()
        return {"name": result[0], "text": result[1]} if result else {"name": f"Document {doc_id}", "text": "No text available"}
    except Exception as e:
        st.error(f"Error fetching subtitle text: {e}")
        return {"name": f"Document {doc_id}", "text": "Error retrieving text"}


def display_search_results(results, db_path):
    """Show the top search results with expanders for detailed content."""
    if not results:
        st.warning("No matching results found.")
        return

    st.header("Matching Subtitles")

    for index, (doc_id, similarity_score) in enumerate(results):
        with st.expander(f"Result {index + 1}: Similarity {similarity_score:.4f}"):
            doc_info = retrieve_subtitle_text(doc_id, db_path)
            st.subheader(f"Video: {doc_info['name']}")

            preview_len = 500
            preview_text = doc_info["text"][:preview_len] + "..." if len(doc_info["text"]) > preview_len else doc_info["text"]
            st.write(preview_text)

            if len(doc_info["text"]) > preview_len:
                if st.button(f"View Full Subtitle - Result {index + 1}", key=f"full_{index}"):
                    st.text_area("", doc_info["text"], height=300)


# Main Function
def app():
    st.title("🎬 Subtitle Search Tool")
    st.markdown("""
    Search for subtitles by providing an audio clip from a movie/TV show or by typing a text query.
    This tool uses semantic search for the best results.
    """)

    # Load configuration
    settings = load_configuration()
    if settings is None:
        return

    # Set up Google API configuration
    setup_google_api(settings)

    # Sidebar settings
    with st.sidebar:
        st.header("Search Configuration")

        # Check if necessary files exist
        db_found = os.path.exists(settings["subtitle_db"])
        chroma_db_found = os.path.exists(settings["vector_db_path"])

        if not db_found or not chroma_db_found:
            st.error("⚠️ Missing database files!")
            st.info("""
            Ensure you have run the subtitle processing pipeline:
            ```
            python subtitle_processing_pipeline.py --stage all
            ```
            """)

        # Configure search options
        result_count = st.slider("Number of Results", min_value=1, max_value=settings["max_display_results"], value=settings["default_results_count"])
        timeout_seconds = st.slider("Timeout (seconds)", min_value=5, max_value=60, value=settings["search_timeout_seconds"])

    # Main content area for input
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Record Audio Query")
        st.write("Speak a line from your favorite movie/TV show:")
        recorded_audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="recorder")
        audio_data = recorded_audio["bytes"] if recorded_audio and "bytes" in recorded_audio else None

    with col2:
        st.header("Or Enter Text Query")
        text_query = st.text_input("Type your search query", "")

    # Process query based on input
    query_text = None
    if audio_data:
        if not settings.get("google_api_key"):
            st.error("Google API Key is not configured. Please enter it in the sidebar.")
        else:
            audio_file = io.BytesIO(audio_data)
            query_text = transcribe_audio_clip(audio_file)
            if query_text:
                st.success(f"Transcription: '{query_text[:100]}...'")
            else:
                st.error("Failed to transcribe audio. Please try again or use a text query.")
    elif text_query:
        query_text = clean_text(text_query)
        st.info(f"Searching for: '{query_text[:100]}...'")

    # Perform the search if a query is provided
    if query_text:
        search_results = execute_search(query_text, settings["vector_db_path"], top_k=result_count, timeout=timeout_seconds)
        display_search_results(search_results, settings["subtitle_db"])


# Run the application
if __name__ == "__main__":
    app()
