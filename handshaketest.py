import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_oracledb import OracleVS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: High-Precision Hybrid Search")

with st.sidebar:
    st.header("Engine Settings")
    available_models = [
        "Gemini 3 Flash (Direct Google)", 
        "Gemini 2.5 Pro (Direct Google)", 
        "Qwen 3 Max Thinking (Alibaba)",
        "Groq Compound (Router Model)",
        "Llama 3.3 70B (Direct Groq)"
    ]
    model_choice = st.selectbox("Select AI Engine:", options=available_models, key="model_v5")
    st.divider()
    st.info("💡 Keyword Search is now hardcoded to prioritize exact 'AI Chatbot' and 'LangChain' matches.")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        conn = oracledb.connect(
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            dsn=st.secrets["DB_DSN"],
            disable_oob=True
        )
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
        
        if "Qwen" in engine_choice:
            llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif "Gemini" in engine_choice:
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

        v_store = OracleVS(client=conn, table_name="RESUME_SEARCH", embedding_function=embeddings)
        return conn, v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}"); st.stop()

conn, v_store, llm = init_connections(model_choice)

# --- 3. Enhanced Hybrid Retrieval ---
def get_targeted_context(query, v_store, conn):
    # 1. Semantic search for "Meaning"
    semantic_docs = v_store.similarity_search(query, k=3)
    
    # 2. SQL LIKE search for "Exact Proof"
    keyword_docs = []
    try:
        cursor = conn.cursor()
        # Search for synonyms of chatbots/ai in case the prompt is broad
        search_terms = ["%chatbot%", "%AI%", "%LangChain%", "%RAG%"]
        for term in search_terms:
            cursor.execute("SELECT TEXT FROM RESUME_SEARCH WHERE TEXT LIKE :1 FETCH FIRST 1 ROWS ONLY", [term])
            for row in cursor:
                keyword_docs.append(Document(page_content=row[0], metadata={"source": "database_keyword"}))
    except:
        pass
    
    return keyword_docs + semantic_docs

# --- 4. Main Interaction ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about Freddy's AI projects or technical leadership."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Does Freddy know AI Chatbots?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving project evidence..."):
            docs = get_targeted_context(prompt, v_store, conn)
            
            # Formatting the context as numbered evidence
            context_list = [f"EVIDENCE {i+1}: {doc.page_content}" for i, doc in enumerate(docs)]
            full_context = "\n\n".join(context_list)
            
            # The Prompt: Instructing the LLM to use the evidence found
            system_prompt = f"""
            SYSTEM: You are Freddy's Lead Recruiter. You must answer the question using ONLY the provided evidence.
            If the evidence mentions specific frameworks like LangChain or Oracle 23ai, mention them.
            
            CONTEXT FROM FREDDY'S DATABASE:
            {full_context}
            
            QUESTION: {prompt}
            """
            
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
