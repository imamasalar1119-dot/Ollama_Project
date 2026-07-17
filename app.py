import os
import sys
import warnings
import json
import uuid
import requests
import streamlit as st

# Suppress harmless deprecation/connection-reset warnings in the console
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    page_title="My ChatGPT",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS - ChatGPT-style look
# ==========================================================

st.markdown(
    """
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Overall font */
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    /* Center the chat column like ChatGPT */
    .block-container {
        max-width: 820px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        margin: 0 auto;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        padding: 0.75rem 0;
        border: none;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #d9d9e3;
        background-color: #ffffff;
        text-align: left;
        font-size: 0.9rem;
        padding: 0.5rem 0.75rem;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #ececf1;
        border-color: #d9d9e3;
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto;
    }

    /* Titles */
    h1, h2, h3 {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# PROJECT DIRECTORIES
# ==========================================================

UPLOAD_FOLDER = "uploaded_files"
CHROMA_DB_DIR = os.path.abspath("chroma_db")
CHAT_HISTORY_DIR = "chat_history"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "history.json")

MODEL_NAME = "qwen2.5:3b"
EMBED_MODEL_NAME = "nomic-embed-text"

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
        json.dump(messages, file, indent=4, ensure_ascii=False)


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

if "document_name" not in st.session_state:
    st.session_state.document_name = None


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


def split_document(document_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(document_text)


def store_embeddings(chunks):
    if len(chunks) == 0:
        st.error("No readable text found in document.")
        return None

    client = get_chroma_client()

    # Delete the old collection (no file lock issue, this is a DB-level delete only)
    if st.session_state.collection_name:
        try:
            client.delete_collection(st.session_state.collection_name)
        except Exception:
            pass

    new_collection_name = f"doc_{uuid.uuid4().hex[:8]}"

    embedding_model = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    try:
        vector_db = Chroma(
            client=client,
            collection_name=new_collection_name,
            embedding_function=embedding_model
        )
        vector_db.add_texts(texts=chunks)

    except Exception as e:
        st.error(
            "❌ ChromaDB could not initialize."
            "This is usually caused by a chromadb / langchain-chroma version mismatch.\n\n"
            "Run this in the terminal:\n"
            "pip install --upgrade chromadb langchain-chroma\n\n"
            f"Technical detail: {e}"
        )
        return None

    st.session_state.collection_name = new_collection_name
    return vector_db


def retrieve_documents(question):
    if st.session_state.vector_db is None:
        return []
    return st.session_state.vector_db.similarity_search(question, k=4)


def remove_document():
    if st.session_state.collection_name:
        try:
            get_chroma_client().delete_collection(st.session_state.collection_name)
        except Exception:
            pass

    st.session_state.vector_db = None
    st.session_state.document_loaded = False
    st.session_state.collection_name = None
    st.session_state.document_name = None


def handle_upload(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    document_text = extract_text(file_path)

    if len(document_text.strip()) == 0:
        st.error("❌ No text found in this document.")
        return

    chunks = split_document(document_text)

    with st.spinner("Reading document..."):
        vector_db = store_embeddings(chunks)

    if vector_db is not None:
        st.session_state.vector_db = vector_db
        st.session_state.document_loaded = True
        st.session_state.document_name = uploaded_file.name
        st.toast(f"✅ {uploaded_file.name} ready to chat about", icon="📄")
    else:
        st.error("Failed to process document.")


def stream_ollama_response(final_prompt):
    """Generator that yields tokens as they arrive, for a ChatGPT-style typing effect."""
    with requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": final_prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 300
            }
        },
        timeout=120,
        stream=True
    ) as response:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def build_prompt(user_question, context):
    if context.strip():
        return f"""You are a helpful AI assistant.

Answer the question using the information provided in the document context below.

If the document context does not contain the answer, then answer the question yourself using your own general knowledge instead, and briefly mention that this part is not from the uploaded document.

========================
DOCUMENT CONTEXT
========================

{context}

========================
QUESTION
========================

{user_question}

========================
ANSWER
========================
"""
    else:
        return f"""You are a helpful, friendly AI assistant, similar to ChatGPT.

Answer the user's question directly and helpfully using your own general knowledge. No document has been uploaded, so just have a normal conversation.

========================
QUESTION
========================

{user_question}

========================
ANSWER
========================
"""


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("### 💬 My ChatGPT")

    if st.button("➕  New chat", use_container_width=True):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()

    st.markdown("")

    with st.expander("📎  Attach a document", expanded=not st.session_state.document_loaded):
        uploaded_file = st.file_uploader(
            "PDF, DOCX or TXT",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None and uploaded_file.name != st.session_state.document_name:
            handle_upload(uploaded_file)
            st.rerun()

    if st.session_state.document_loaded:
        st.success(f"📄 {st.session_state.document_name}")
        if st.button("🗑️  Remove document", use_container_width=True):
            remove_document()
            st.rerun()

    st.markdown("---")
    st.markdown("##### Recent messages")

    if len(st.session_state.messages) == 0:
        st.caption("No messages yet.")
    else:
        for message in st.session_state.messages[-8:]:
            role_icon = "🧑" if message["role"] == "user" else "🤖"
            preview = message["content"][:45].replace("\n", " ")
            st.caption(f"{role_icon} {preview}{'…' if len(message['content']) > 45 else ''}")

    st.markdown("---")

    if st.button("🗑️  Clear chat history", use_container_width=True):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()

    st.caption(f"Model: `{MODEL_NAME}`  \nEmbeddings: `{EMBED_MODEL_NAME}`")


# ==========================================================
# MAIN CHAT AREA
# ==========================================================

if len(st.session_state.messages) == 0:
    st.markdown(
        "<h2 style='text-align:center; margin-top:15vh; color:#40414f;'>"
        "What can I help with?</h2>",
        unsafe_allow_html=True
    )

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "💬"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input("Message My ChatGPT...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="💬"):

        try:
            context = ""
            if st.session_state.document_loaded:
                docs = retrieve_documents(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = build_prompt(prompt, context)

            answer = st.write_stream(stream_ollama_response(final_prompt))

            st.session_state.messages.append({"role": "assistant", "content": answer})
            save_chat_history(st.session_state.messages)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the Ollama server. "
                "Keep `ollama serve` running in a separate terminal, then try again."
            )

        except Exception as e:
            st.error(f"Error: {e}")