import os
import requests
import streamlit as st
from pypdf import PdfReader
from docx import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Offline Document Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Offline Document Chatbot")
st.write("Powered by Ollama")

# ---------------------------------------------------
# Project Folders
# ---------------------------------------------------

UPLOAD_FOLDER = "uploaded_files"
CHROMA_DB_DIR = "chroma_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------
# Chat History
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# Extract Text
# ---------------------------------------------------

def extract_text(file_path):

    extension = file_path.split(".")[-1].lower()

    text = ""

    if extension == "pdf":

        reader = PdfReader(file_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    elif extension == "docx":

        doc = Document(file_path)

        for para in doc.paragraphs:
            text += para.text + "\n"

    elif extension == "txt":

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    return text


# ---------------------------------------------------
# Split Document
# ---------------------------------------------------

def split_document(document_text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(document_text)

    return chunks


# ---------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------

def generate_embeddings(chunks):

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectors = embedding_model.embed_documents(chunks)

    return vectors


# ---------------------------------------------------
# Store Embeddings in ChromaDB
# ---------------------------------------------------

def store_embeddings(chunks):

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    return vector_db
    # ---------------------------------------------------
# Upload Section
# ---------------------------------------------------

st.header("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX or TXT",
    type=["pdf", "docx", "txt"]
)

# Vector database object
vector_db = None

if uploaded_file:

    # -----------------------------
    # Save Uploaded File
    # -----------------------------

    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

    # -----------------------------
    # Extract Text
    # -----------------------------

    document_text = extract_text(file_path)

    st.subheader("📖 Extracted Text")

    st.text_area(
        "Document Content",
        document_text,
        height=250
    )

    # -----------------------------
    # Split into Chunks
    # -----------------------------

    chunks = split_document(document_text)

    st.success(f"✅ Total Chunks Created: {len(chunks)}")

    with st.expander("View Document Chunks"):

        for i, chunk in enumerate(chunks):

            st.markdown(f"### Chunk {i + 1}")
            st.write(chunk)

    # -----------------------------
    # Generate & Store Embeddings
    # -----------------------------

    with st.spinner("Generating embeddings and storing in ChromaDB..."):

        vector_db = store_embeddings(chunks)

    st.success("✅ Embeddings stored successfully in ChromaDB.")

    st.info(f"📦 {len(chunks)} chunks stored in the vector database.")
    # ---------------------------------------------------
# Chatbot
# ---------------------------------------------------

st.header("💬 Chat")

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask anything...")

if prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": "qwen2.5:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 150
                        }
                    },
                    timeout=120
                )

                answer = response.json()["response"]

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")