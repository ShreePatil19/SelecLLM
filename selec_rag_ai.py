import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()

# -------------------- CONFIG --------------------
DATA_DIR = "data"
DB_FAISS_PATH = "faiss_db"
PASSWORD = st.secrets["AUTH_PASSWORD"] if "AUTH_PASSWORD" in st.secrets else "admin"

# -------------------- AUTH --------------------
if "auth_passed" not in st.session_state:
    st.session_state["auth_passed"] = False

if not st.session_state["auth_passed"]:
    pwd = st.text_input("Enter password", type="password")
    if pwd == PASSWORD:
        st.session_state["auth_passed"] = True
        st.experimental_rerun()
    else:
        st.stop()

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_phi2():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_phi2()

# -------------------- PDF HANDLING --------------------
def save_uploaded_files(uploaded_files):
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out_file:
            out_file.write(f.read())

# -------------------- VECTOR DB --------------------
@st.cache_resource
def load_vectorstore():
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings())
    else:
        return None

def ingest_docs():
    loaders = [PyPDFLoader(os.path.join(DATA_DIR, file)) for file in os.listdir(DATA_DIR)]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

# -------------------- INIT --------------------
st.title("📄 Selec AI Assistant")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    save_uploaded_files(uploaded_files)
    db = ingest_docs()
    st.success("Documents ingested and vectorstore created.")
else:
    db = load_vectorstore()

if db is None:
    st.warning("Please upload PDF(s) to start.")
    st.stop()

# -------------------- CONVERSATION --------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever(), memory=st.session_state.memory)

query = st.chat_input("Ask a question...")
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.chat_message("assistant").write(response)

# -------------------- HISTORY --------------------
with st.expander("🧠 Memory Log"):
    for msg in st.session_state.memory.chat_memory.messages:
        st.markdown(f"**{msg.type.capitalize()}:** {msg.content}")
