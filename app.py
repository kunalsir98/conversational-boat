# ===============================
# app.py
# ===============================

import os
import streamlit as st
import bs4
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ===============================
# Load environment variables
# ===============================
load_dotenv()

# ===============================
# Streamlit page config
# ===============================
st.set_page_config(
    page_title="Diffusion Models RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Diffusion Models Chatbot")
st.caption("LangChain 1.x · Groq · Chroma · HuggingFace · History-Aware RAG")

# ===============================
# Cache heavy resources
# ===============================
@st.cache_resource
def load_rag_chain():
    # -------- LLM (Groq) --------
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # -------- Embeddings --------
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # -------- Load documents --------
    loader = WebBaseLoader(
        web_paths=(
            "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()

    # -------- Split documents --------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)

    # -------- Vector store --------
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # -------- Prompts --------
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    # -------- Question rewriter --------
    question_rewriter = (
        contextualize_q_prompt
        | model
        | StrOutputParser()
    )

    # -------- History-aware retriever --------
    history_aware_retriever = RunnableLambda(
        lambda x: retriever.invoke(
            question_rewriter.invoke(x)
        )
    )

    # -------- Final RAG chain --------
    rag_chain = (
        {
            "context": history_aware_retriever,
            "input": RunnablePassthrough()
        }
        | qa_prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


# ===============================
# Initialize RAG chain
# ===============================
rag_chain = load_rag_chain()

# ===============================
# Chat history (Streamlit state)
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# Display chat messages
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# Chat input
# ===============================
user_input = st.chat_input("Ask a question about diffusion models...")

if user_input:
    # Show user message
    st.chat_message("human").markdown(user_input)
    st.session_state.messages.append(
        {"role": "human", "content": user_input}
    )

    # Invoke RAG chain
    response = rag_chain.invoke(
        {
            "input": user_input,
            "chat_history": st.session_state.chat_history
        }
    )

    # Show assistant message
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Update chat history (for context)
    st.session_state.chat_history.extend(
        [
            {"role": "human", "content": user_input},
            {"role": "assistant", "content": response},
        ]
    )
