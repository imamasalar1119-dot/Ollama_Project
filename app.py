import os
import json
import uuid
import shutil
import requests
import streamlit as st

from pypdf import PdfReader
from docx import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Offline Document ChatGPT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Offline Document ChatGPT")
st.caption("Powered by Ollama + ChromaDB")

# ==========================================================
# PROJECT FOLDERS
# ==========================================================

UPLOAD_FOLDER = "uploaded_files"
CHROMA_DB_DIR = "chroma_db"
CHAT_HISTORY_DIR = "chat_history"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ==========================================================
# PROJECT DIRECTORIES
# ==========================================================

CHROMA_DB_DIR = os.path.abspath("chroma_db")
CHAT_HISTORY_DIR = "chat_history"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# ==========================================================
# CHAT HISTORY
# ==========================================================

HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "history.json")

# ==========================================================
# PERSISTENT CHROMA CLIENT (created once, reused everywhere)
# ==========================================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def load_chat_history():

    if os.path.exists(HISTORY_FILE):

        with open(HISTORY_FILE, "r", encoding="utf-8") as file:

            return json.load(file)

    return []


def save_chat_history(messages):

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:

        json.dump(
            messages,
            file,
            indent=4,
            ensure_ascii=False
        )

# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = load_chat_history()

if "vector_db" not in st.session_state:

    st.session_state.vector_db = None

if "document_loaded" not in st.session_state:

    st.session_state.document_loaded = False

if "collection_name" not in st.session_state:

    st.session_state.collection_name = None

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📂 Document Chat")

    if st.button("🆕 New Chat"):

        st.session_state.messages = []
        save_chat_history([])
        st.rerun()

    st.divider()

    st.write("### Previous Messages")

    if len(st.session_state.messages) == 0:

        st.info("No chat history.")

    else:

        for message in st.session_state.messages[-10:]:

            role = "🧑" if message["role"] == "user" else "🤖"

            st.write(f"{role} {message['content'][:40]}...")
            # ==========================================================
# TEXT EXTRACTION
# ==========================================================

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


# ==========================================================
# SPLIT DOCUMENT
# ==========================================================

def split_document(document_text):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=100

    )

    chunks = splitter.split_text(document_text)

    return chunks


# ==========================================================
# STORE EMBEDDINGS IN CHROMADB
# ==========================================================

def store_embeddings(chunks):

    if len(chunks) == 0:

        st.error("No readable text found in document.")

        return None

    client = get_chroma_client()

    # Purani collection delete (file lock nahi lagta, sirf DB-level delete hai)
    if st.session_state.collection_name:

        try:
            client.delete_collection(st.session_state.collection_name)
        except Exception:
            pass

    new_collection_name = f"doc_{uuid.uuid4().hex[:8]}"

    embedding_model = OllamaEmbeddings(

        model="nomic-embed-text"

    )

    try:

        vector_db = Chroma(

            client=client,

            collection_name=new_collection_name,

            embedding_function=embedding_model

        )

        vector_db.add_texts(texts=chunks)

    except Exception as e:

        st.error(
            "❌ ChromaDB could not initialize. Ye usually chromadb aur "
            "langchain-chroma ke version mismatch ki wajah se hota hai.\n\n"
            "Terminal me ye run karein:\n"
            "pip install --upgrade chromadb langchain-chroma\n\n"
            f"Technical detail: {e}"
        )

        return None

    st.session_state.collection_name = new_collection_name

    return vector_db


# ==========================================================
# RETRIEVE RELEVANT CHUNKS
# ==========================================================

def retrieve_documents(question):

    if st.session_state.vector_db is None:

        return []

    documents = st.session_state.vector_db.similarity_search(

        question,

        k=4

    )

    return documents
    # ==========================================================
# DOCUMENT UPLOAD
# ==========================================================

st.header("📄 Upload Document")

uploaded_file = st.file_uploader(

    "Upload PDF, DOCX or TXT",

    type=["pdf", "docx", "txt"]

)

if uploaded_file is not None:

    file_path = os.path.join(

        UPLOAD_FOLDER,

        uploaded_file.name

    )

    with open(file_path, "wb") as file:

        file.write(uploaded_file.getbuffer())

    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

    # ---------------------------------------
    # Extract Text
    # ---------------------------------------

    document_text = extract_text(file_path)

    if len(document_text.strip()) == 0:

        st.error("❌ No text found in this document.")

        st.stop()

    st.subheader("📄 Extracted Text")

    st.text_area(

        "Preview",

        document_text,

        height=250

    )

    # ---------------------------------------
    # Split into Chunks
    # ---------------------------------------

    chunks = split_document(document_text)

    st.success(f"✅ Total Chunks: {len(chunks)}")

    with st.expander("View Chunks"):

        for i, chunk in enumerate(chunks):

            st.markdown(f"### Chunk {i+1}")

            st.write(chunk)

    # ---------------------------------------
    # Store Embeddings
    # ---------------------------------------

    with st.spinner("Generating Embeddings..."):

        vector_db = store_embeddings(chunks)

    if vector_db is not None:

        st.session_state.vector_db = vector_db

        st.session_state.document_loaded = True

        st.success("✅ ChromaDB Ready")

    else:

        st.error("Failed to create vector database.")
        # ==========================================================
# CHATBOT (RAG + OLLAMA)
# ==========================================================

st.header("💬 Chat with your Document")

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your document...")

if prompt:

    # ---------------- User Message ----------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    save_chat_history(st.session_state.messages)

    with st.chat_message("user"):

        st.markdown(prompt)

    # ---------------- Assistant ----------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # ---------- Retrieve Relevant Chunks ----------

                context = ""

                if st.session_state.document_loaded:

                    docs = retrieve_documents(prompt)

                    context = "\n\n".join(
                        [doc.page_content for doc in docs]
                    )

                # ---------- Final Prompt ----------

                final_prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information provided in the document context.

If the answer is not available in the document, reply exactly:

'I could not find the answer in the uploaded document.'

========================
DOCUMENT CONTEXT
========================

{context}

========================
QUESTION
========================

{prompt}

========================
ANSWER
========================
"""

                response = requests.post(

                    "http://127.0.0.1:11434/api/generate",

                    json={

                        "model": "qwen2.5:3b",

                        "prompt": final_prompt,

                        "stream": False,

                        "options": {

                            "temperature": 0.2,

                            "num_predict": 300

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

                save_chat_history(st.session_state.messages)

            except Exception as e:

                st.error(f"Error: {e}")
                # ==========================================================
# SIDEBAR UTILITIES
# ==========================================================

with st.sidebar:

    st.divider()

    st.subheader("📊 Project Status")

    st.write(
        "Document Loaded:",
        "✅ Yes" if st.session_state.document_loaded else "❌ No"
    )

    st.write(
        "Messages:",
        len(st.session_state.messages)
    )

    st.divider()

    # ----------------------------
    # Clear Chat
    # ----------------------------

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        save_chat_history([])

        st.success("Chat history cleared.")

        st.rerun()

    # ----------------------------
    # Delete Vector Database
    # ----------------------------

    if st.button("❌ Remove Document"):

        if st.session_state.collection_name:

            try:
                get_chroma_client().delete_collection(st.session_state.collection_name)
            except Exception:
                pass

        st.session_state.vector_db = None
        st.session_state.document_loaded = False
        st.session_state.collection_name = None

        st.success("Document removed successfully.")

        st.rerun()

    st.divider()

    st.info(
        """
### Supported Files

- PDF
- DOCX
- TXT

### AI Models

- Qwen2.5:3B
- nomic-embed-text

### Vector Database

- ChromaDB
"""
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Offline Document ChatGPT | "
    "Streamlit + Ollama + ChromaDB + LangChain"
)