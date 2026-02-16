import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")
st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Zilliz Cloud Hybrid Search")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    # Model Selection Logic
    if "Qwen" in engine_choice:
        llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    elif "Gemini" in engine_choice:
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

    # Milvus Connection
    v_store = Milvus(
        embedding_function=embeddings,
        collection_name="RESUME_SEARCH",
        connection_args={
            "uri": st.secrets["ZILLIZ_URI"],
            "token": st.secrets["ZILLIZ_TOKEN"],
            "secure": True,
            "pool_size": 5,
            "client_config": {"wait_for_ready": True}
        }
    )
    return v_store, llm

# Initialize based on sidebar selection (omitted sidebar code for brevity)
v_store, llm = init_connections(st.sidebar.selectbox("Model", ["Gemini 3 Flash (Direct Google)", "Llama 3.3 70B (Direct Groq)"]))

# --- 3. Chat Interaction ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy's AI experience"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Find this section in your script ---
    with st.chat_message("assistant"):
        with st.spinner("Retrieving project evidence..."):
            docs = get_targeted_context(prompt, v_store, conn)
            
            # ... (formatting context code) ...
            
            # 🟢 THE UPDATED PART:
            response = llm.invoke(system_prompt)
            
            # We only want the TEXT, not the signatures or extras
            clean_answer = response.content 
    
            st.markdown(clean_answer)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})
