import sqlite3
import io
import zipfile
import re
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure required NLTK resources are downloaded
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

# Functions to process subtitles
def extract_text_from_zip(zip_data):
    """Extract text from a zip file containing subtitles."""
    try:
        with io.BytesIO(zip_data) as file:
            with zipfile.ZipFile(file, "r") as zip_file:
                content = zip_file.read(zip_file.namelist()[0])
    except Exception as e:
        print(f"Error: {e}")
        return None
    return content.decode('latin-1')

def clean_subtitle_text(subtitle_text):
    """Remove timestamps, indices, and unnecessary characters from subtitle text."""
    if not subtitle_text:
        return ""
    
    lines = subtitle_text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip() and not line.isdigit() and not re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line)]
    return " ".join(cleaned_lines)

def remove_special_chars(text):
    """Remove special characters except letters and spaces."""
    return re.sub(r'[^a-zA-Z0-9\s]', "", text)

def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words])

# Stream subtitle data and process it in batches
def process_and_store_subtitles(source_db, output_db, batch_size=100):
    """Stream subtitle data from the database and store processed data."""
    try:
        conn = sqlite3.connect(source_db)
        cursor = conn.cursor()
        output_conn = sqlite3.connect(output_db)
        output_cursor = output_conn.cursor()

        output_cursor.execute("DROP TABLE IF EXISTS processed_subtitles")
        output_cursor.execute("""
            CREATE TABLE processed_subtitles (
                id INTEGER PRIMARY KEY,
                name TEXT,
                original_content TEXT,
                extracted_text TEXT,
                cleaned_text TEXT,
                no_special_text TEXT
            )
        """)
        output_conn.commit()

        cursor.execute("SELECT COUNT(*) FROM zipfiles")
        total_rows = cursor.fetchone()[0]
        print(f"Total rows to process: {total_rows}")

        offset = 0
        while True:
            cursor.execute("SELECT id, name, content FROM zipfiles LIMIT ? OFFSET ?", (batch_size, offset))
            rows = cursor.fetchall()
            if not rows:
                break
            
            batch_data = []
            for row in rows:
                id, name, content = row
                extracted_text = extract_text_from_zip(content)
                cleaned_text = clean_subtitle_text(extracted_text)
                no_special_text = remove_special_chars(cleaned_text)
                batch_data.append((id, name, content, extracted_text, cleaned_text, no_special_text))
            
            output_cursor.executemany("""
                INSERT INTO processed_subtitles (id, name, original_content, extracted_text, cleaned_text, no_special_text)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch_data)
            output_conn.commit()

            offset += batch_size
            print(f"Processed {offset}/{total_rows} rows")

        conn.close()
        output_conn.close()
        print("Data processing completed.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Compute and store TF-IDF values
def compute_tfidf(source_db, tfidf_db, batch_size=1000):
    """Compute TF-IDF vectors and store them in a database."""
    try:
        conn = sqlite3.connect(source_db)
        cursor = conn.cursor()

        tfidf_conn = sqlite3.connect(tfidf_db)
        tfidf_cursor = tfidf_conn.cursor()

        tfidf_cursor.execute("DROP TABLE IF EXISTS tfidf_subtitles")
        tfidf_cursor.execute("""
            CREATE TABLE tfidf_subtitles (
                id INTEGER PRIMARY KEY,
                indices TEXT,
                tfidf_values TEXT
            )
        """)
        tfidf_conn.commit()

        vectorizer = TfidfVectorizer()
        offset = 0
        cursor.execute("SELECT COUNT(*) FROM processed_subtitles")
        total_rows = cursor.fetchone()[0]

        while True:
            cursor.execute("SELECT id, no_special_text FROM processed_subtitles LIMIT ? OFFSET ?", (batch_size, offset))
            rows = cursor.fetchall()
            if not rows:
                break
            
            ids, texts = zip(*rows)
            processed_texts = [preprocess_text(text) for text in texts]
            tfidf_matrix = vectorizer.fit_transform(processed_texts)

            batch_data = []
            for id, vector in zip(ids, tfidf_matrix):
                indices = " ".join(map(str, vector.indices))
                tfidf_values = " ".join(map(str, vector.data))
                batch_data.append((id, indices, tfidf_values))

            tfidf_cursor.executemany("""
                INSERT INTO tfidf_subtitles (id, indices, tfidf_values)
                VALUES (?, ?, ?)
            """, batch_data)
            tfidf_conn.commit()

            offset += batch_size
            print(f"Processed {offset}/{total_rows} rows")

        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        conn.close()
        tfidf_conn.close()
        print("TF-IDF computation completed.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Compute and store embeddings in ChromaDB
def compute_embeddings_and_store(source_db, chroma_path, metadata_db, batch_size=500, resume_from=0):
    """Compute semantic embeddings and store them in ChromaDB."""
    try:
        conn = sqlite3.connect(source_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM processed_subtitles")
        total_rows = cursor.fetchone()[0]
        print(f"Total rows to process: {total_rows}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        vector_store = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)

        metadata_conn = sqlite3.connect(metadata_db)
        metadata_cursor = metadata_conn.cursor()

        metadata_cursor.execute("DROP TABLE IF EXISTS chroma_metadata")
        metadata_cursor.execute("""
            CREATE TABLE chroma_metadata (
                chroma_id TEXT PRIMARY KEY,
                subtitle_id INTEGER,
                chunk TEXT
            )
        """)
        metadata_conn.commit()

        offset = resume_from
        while offset < total_rows:
            cursor.execute("SELECT id, cleaned_text FROM processed_subtitles LIMIT ? OFFSET ?", (batch_size, offset))
            rows = cursor.fetchall()
            if not rows:
                break
            
            batch_texts, batch_metadata, ids = [], [], []
            for id, text in rows:
                if text:
                    chunks = text_splitter.split_text(text)
                    for idx, chunk in enumerate(chunks):
                        batch_texts.append(chunk)
                        batch_metadata.append({"num": id, "chunk": chunk})
                        ids.append(f"{id}_{idx}")

            embeddings = embedding_model.embed_documents(batch_texts)
            vector_store.add_texts(texts=batch_texts, metadatas=batch_metadata, ids=ids)

            metadata_records = [(ids[i], batch_metadata[i]["num"], batch_metadata[i]["chunk"]) for i in range(len(batch_metadata))]
            metadata_cursor.executemany("""
                INSERT INTO chroma_metadata (chroma_id, subtitle_id, chunk)
                VALUES (?, ?, ?)
            """, metadata_records)
            metadata_conn.commit()

            offset += batch_size
            print(f"Processed {offset}/{total_rows} rows")

        vector_store.persist()
        conn.close()
        metadata_conn.close()
        print("Embeddings stored in ChromaDB.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Main function to run the entire pipeline
def main():
    # Adjust the paths and options as needed
    source_db = "source_subtitles.db"
    processed_db = "processed_data.db"
    tfidf_db = "tfidf_data.db"
    chroma_path = "chroma_vectors"
    metadata_db = "chroma_metadata.db"

    stage = "all"  # Choose between "process", "verify", "tfidf", "embeddings", or "all"

    if stage == "process" or stage == "all":
        print("Processing subtitles...")
        process_and_store_subtitles(source_db, processed_db)

    if stage == "tfidf" or stage == "all":
        print("Computing TF-IDF vectors...")
        compute_tfidf(processed_db, tfidf_db)

    if stage == "embeddings" or stage == "all":
        print("Computing embeddings...")
        compute_embeddings_and_store(processed_db, chroma_path, metadata_db)

    print("Pipeline completed.")

if __name__ == "__main__":
    main()
