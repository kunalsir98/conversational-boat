import os
import streamlit as st
from dotenv import load_dotenv
import bs4

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===============================
# Load environment variables
# ===============================
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ===============================
# Streamlit page config
# ===============================
st.set_page_config(
    page_title="Death Note GPT",
    page_icon="🧠",
    layout="centered"
)

# ===============================
# Custom CSS (🔥 UI MAGIC)
# ===============================
st.markdown(
    """
    <style>
        body {
            background-color: #0f172a;
        }
        .main-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7c3aed, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2em;
        }
        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2em;
        }
        .stChatMessage[data-testid="chat-message-user"] {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 10px;
        }
        .stChatMessage[data-testid="chat-message-assistant"] {
            background-color: #020617;
            border-radius: 12px;
            padding: 10px;
        }
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Header
# ===============================
st.markdown('<div class="main-title">🧠 "Death Note GPT"</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask deep questions. Get intelligent answers.</div>',
    unsafe_allow_html=True
)

# ===============================
# Session state
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:

    with st.spinner("🔮 Initializing intelligence engine..."):

        # ===============================
        # LLM
        # ===============================
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # ===============================
        # Embeddings
        # ===============================
        embeddings = HuggingFaceEmbeddings(
            model="all-MiniLM-L6-v2"
        )

        # ===============================
        # Load content
        # ===============================
        loader = WebBaseLoader(
            web_paths=("https://en.wikipedia.org/wiki/Death_Note",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(id="mw-content-text")
            ),
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # ===============================
        # Prompt
        # ===============================
        system_prompt = (
            "You are an intelligent anime expert assistant. "
            "Use the provided context to answer clearly and concisely. "
            "If the answer is unknown, say so honestly. "
            "Limit responses to three sentences.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )

        rag_chain = (
            {
                "context": retriever,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state.rag_chain = rag_chain

# ===============================
# Display messages
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# Chat input
# ===============================
user_input = st.chat_input("Ask about characters, story, themes...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            response = st.session_state.rag_chain.invoke(user_input)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Controls</div>', unsafe_allow_html=True)
    st.markdown("")

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Model:** LLaMA-3.3 (Groq)")
    st.markdown("**Mode:** Intelligent RAG Chat")
