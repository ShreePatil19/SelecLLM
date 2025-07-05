import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import tempfile
from auth import register_user, authenticate_user
from chat_store import load_history, save_message, get_chat_titles, get_chat_by_id

# ----------------------------
# AUTHENTICATION
# ----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.active_chat = "New Chat"

if not st.session_state.authenticated:
    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔐 Login or Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("Registered successfully! You can now log in.")
            else:
                st.error("User already exists")

    st.stop()

# ----------------------------
# MAIN APP
# ----------------------------
st.set_page_config(page_title="🤖 Selec LLM Assistant", layout="wide")
st.title("🤖 Selec LLM Assistant")

# Sidebar
st.sidebar.title(f"👤 {st.session_state.username}")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Just GPT", "GPT + Your Docs"])
model_choice = st.sidebar.selectbox("Choose Model", ["phi3:mini", "llama3", "mistral", "flan-t5-small"])
chat_titles = ["New Chat"] + get_chat_titles(st.session_state.username)
selected_chat = st.sidebar.selectbox("💬 Your Chats", chat_titles)

if selected_chat != st.session_state.get("active_chat"):
    st.session_state.active_chat = selected_chat

# Load model
@st.cache_resource
def load_model(model_choice):
    if model_choice == "flan-t5-small":
        model = AutoModelForCausalLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)
    else:
        return Ollama(model=model_choice)

llm = load_model(model_choice)

@st.cache_resource
def embed_chunks(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(_chunks, embeddings)
    return db.as_retriever()

qa_chain = None
if mode == "GPT + Your Docs":
    pdf = st.file_uploader("📄 Upload PDF", type="pdf")
    if pdf:
        with st.spinner("🔍 Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            retriever = embed_chunks(chunks)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Load current chat history
_, history = get_chat_by_id(st.session_state.username, st.session_state.active_chat)

# Show chat history
if history:
    for i, (q, r) in enumerate(history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(r)

# Chat input
query = st.chat_input("Ask something...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                if mode == "Just GPT":
                    response = llm(query)
                else:
                    if pdf:
                        response = qa_chain.run(query)
                    else:
                        st.error("❗Please upload a PDF.")
                        st.stop()
                st.markdown(response)
                # Save to chat history
                chat_title = save_message(st.session_state.username, st.session_state.active_chat, query, response)
                st.session_state.active_chat = chat_title
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
