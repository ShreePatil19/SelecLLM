import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Setup
st.set_page_config(page_title="Selec AI Assistant", layout="wide")
UPLOAD_FOLDER = "uploads"
DB_FOLDER = "vectorstore"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Load the FLAN-T5 model
def load_llm():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# Load vector DB if exists
def load_vector_db():
    index_path = os.path.join(DB_FOLDER, "index.faiss")
    if os.path.exists(index_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(DB_FOLDER, embeddings, allow_dangerous_deserialization=True)
    return None

# Process uploaded PDFs
def process_pdfs(uploaded_files):
    documents = []
    for file in uploaded_files:
        path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FOLDER)
    return db

# Sidebar
st.sidebar.title("📂 PDF Upload")
uploaded_files = st.sidebar.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

# Process or Load Vector DB
if uploaded_files:
    db = process_pdfs(uploaded_files)
else:
    db = load_vector_db()

retriever = db.as_retriever() if db else None
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever) if retriever else None

# UI
mode = st.selectbox("Select Mode", ["💬 Chat with GPT", "📄 Ask from PDFs"])
query = st.text_input("Enter your query here:")

if query:
    if mode == "💬 Chat with GPT":
        with st.spinner("Thinking..."):
            result = llm.invoke(query)
            st.success(result)
    elif mode == "📄 Ask from PDFs":
        if qa_chain:
            with st.spinner("Searching PDFs..."):
                answer = qa_chain.run(query)
                st.success(answer)
        else:
            st.error("⚠️ No PDFs found. Please upload them in the sidebar.")
