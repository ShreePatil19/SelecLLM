import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import tempfile
import time

from auth import register_user, authenticate_user
from chat_store import save_message, load_history, get_chat_titles, get_chat_by_id, create_new_chat

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="🧠 Selec LLM Assistant", layout="wide")

# ------------------ Session Defaults ------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

# ------------------ Authentication ------------------
if not st.session_state.authenticated:
    st.title("🔐 Login or Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.chat_id = None
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("✅ Registered! Now log in.")
            else:
                st.error("⚠️ User already exists")
    st.stop()

# ------------------ Sidebar ------------------
st.sidebar.title("🗂️ Your Chats")
chat_titles = get_chat_titles(st.session_state.username)
selected_title = st.sidebar.selectbox("Select Chat", options=chat_titles + ["➕ New Chat"])

if selected_title == "➕ New Chat":
    new_id = create_new_chat(st.session_state.username)
    st.session_state.chat_id = new_id
    st.rerun()
else:
    st.session_state.chat_id = selected_title

if st.sidebar.button("🔁 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.chat_id = None
    st.rerun()

st.sidebar.markdown("---")
mode = st.sidebar.radio("Chat Mode", ("Just GPT", "GPT + Your Docs"))
model_choice = st.sidebar.selectbox("LLM", ["phi3:mini", "flan-t5-small", "llama3", "mistral"])

# ------------------ Load LLM ------------------
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

# ------------------ Embed Chunks ------------------
@st.cache_resource
def embed_chunks(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(_chunks, embeddings)
    return db.as_retriever()

# ------------------ Chat UI ------------------
st.title("🤖 Selec AI Assistant")

if not st.session_state.chat_id:
    st.warning("🛑 Please select or create a chat.")
    st.stop()

chat_history = load_history(st.session_state.username, st.session_state.chat_id)
for msg in chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "ai"):
        st.markdown(msg["message"])

# ------------------ PDF Upload ------------------
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

# ------------------ Typewriter Effect ------------------
def type_writer_effect(message):
    container = st.empty()
    text = ""
    for char in message:
        text += char
        container.markdown(text + "▌")
        time.sleep(0.01)

# ------------------ Chat Input ------------------
query = st.chat_input("Ask something...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    save_message(st.session_state.username, st.session_state.chat_id, "user", query)

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

            save_message(st.session_state.username, st.session_state.chat_id, "ai", response)
            with st.chat_message("ai"):
                type_writer_effect(response)

        except Exception as e:
            st.error(f"❌ Error: {e}")
