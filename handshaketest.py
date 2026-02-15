import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import OracleVS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar & Model Selection ---
# --- 1.1 Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.title("🤖 Freddy's AI Career Assistant")
st.caption("AI enable search powered by Oracle keyword+vector, RAG, Google embedding, Gemini flash 3.0 LLM, Llama 3.3 70B")

with st.sidebar:
    st.header("Engine Settings")
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=[
            "Gemini 3 Flash (Direct Google)", 
            "Gemini 2.5 Pro (Direct Google)",  # New Option
            "Llama 3.3 70B (Direct Groq)", 
            "Llama 3.3 70B (OpenRouter Free)"
        ],
        index=0
    )
    # 2026 Tip: Gemini 2.5 Pro includes a "Thinking" budget
    if "2.5 Pro" in model_choice:
        st.caption("✨ Using Thinking Mode for deep reasoning.")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        conn = oracledb.connect(
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            dsn=st.secrets["DB_DSN"]
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # LLM Logic
        if engine_choice == "Gemini 3 Flash (Direct Google)":
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
        elif engine_choice == "Gemini 2.5 Pro (Direct Google)":
            # Direct Google setup for the Thinking Model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                thinking_budget=1024  # Activates the "Thinking" reasoning steps
            )
        elif engine_choice == "Llama 3.3 70B (Direct Groq)":
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=st.secrets["GROQ_API_KEY"]
            )
        else:
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct:free",
                openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=1000
            )

        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH", 
            embedding_function=embeddings
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections(model_choice)

# --- 3. Chat Session State & UI (Remaining code stays the same) ---
# ... [rest of the chat logic] ...
