import os
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# ✅ FIX 1: Load API key from Streamlit secrets instead of hardcoding

VECTOR_STORE_PATH = "vectorstore"

st.set_page_config(page_title="RAG Chat App", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 600; margin-bottom: 0.2rem; }
    .sub-header { color: gray; font-size: 0.95rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt"])
    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.markdown("**Embeddings:** FastEmbed")
    st.markdown("**Vector Store:** FAISS")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset DB", use_container_width=True):
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            st.session_state.retriever = None
            st.session_state.file_name = None
            st.session_state.messages = []
            st.rerun()
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
- Upload a PDF or TXT file
- Ask questions about your document
- AI remembers your conversation
- Click **Clear Chat** to start over
- Click **Reset DB** to upload a new file
""")

@st.cache_resource
def load_embeddings():
    return FastEmbedEmbeddings()

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def build_vector_store(file_bytes, file_name):
    suffix = ".pdf" if file_name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        temp_path = f.name
    # ✅ FIX 2: Always clean up the temp file using try/finally
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        db = FAISS.from_documents(docs, load_embeddings())
        db.save_local(VECTOR_STORE_PATH)
        return db.as_retriever(search_kwargs={"k": 3})
    finally:
        os.unlink(temp_path)

def load_vector_store():
    db = FAISS.load_local(VECTOR_STORE_PATH, load_embeddings(), allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

if uploaded_file and uploaded_file.name != st.session_state.file_name:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        st.session_state.retriever = build_vector_store(uploaded_file.read(), uploaded_file.name)
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []
    st.sidebar.success(f"✅ {uploaded_file.name} ready!")
elif os.path.exists(VECTOR_STORE_PATH) and st.session_state.retriever is None:
    st.session_state.retriever = load_vector_store()

llm = load_llm()

prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the context and chat history to answer the question accurately.
If the answer is not in the context, say "I don't know based on the document."

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(messages):
    if not messages:
        return "No previous conversation."
    history = ""
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "AI"
        history += f"{role}: {msg['content']}\n"
    return history

st.markdown('<div class="main-header">📄 RAG Chat App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF or TXT file and ask questions about it</div>', unsafe_allow_html=True)

if st.session_state.file_name:
    st.success(f"📂 Active file: **{st.session_state.file_name}** — Ready to chat!")
else:
    st.info("👈 Upload a PDF or TXT file from the sidebar to get started.")

st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask something about your document..."):
    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload a file first using the sidebar!")
    else:
        # ✅ FIX 3: Snapshot history BEFORE appending the new user message
        # to avoid the question appearing twice (once in history, once as {question})
        history_snapshot = format_history(st.session_state.messages)

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = (
                    {
                        "context": st.session_state.retriever | format_docs,
                        "question": RunnablePassthrough(),
                        "chat_history": lambda _: history_snapshot,
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                answer = chain.invoke(query)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})