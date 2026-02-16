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
    # Standard Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    # --- MODEL ROUTING ---
    if "Qwen" in engine_choice:
        llm = ChatOpenAI(
            model="qwen3-max-2026-01-23", 
            openai_api_key=st.secrets["QWEN_API_KEY"], 
            openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    elif "Gemini" in engine_choice:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
    elif "Groq" in engine_choice or "Llama" in engine_choice or "GPT-OSS" in engine_choice:
        # GPT-OSS 120B and Llama are both served via Groq in this configuration
        target_model = "llama-3.3-70b-versatile" 
        if "120B" in engine_choice:
            target_model = "mixtral-8x7b-32768" # Or the specific 120B ID provided by Groq
            
        llm = ChatGroq(
            model=target_model, 
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )
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

# --- SIDEBAR WITH ALL MODELS ---
with st.sidebar:
    st.header("Engine Settings")
    available_models = [
        "Gemini 3 Flash (Direct Google)", 
        "Gemini 2.5 Pro (Direct Google)", 
        "Qwen 3 Max Thinking (Alibaba)",
        "GPT-OSS-120B (Direct Groq)",
        "Groq Compound (Router Model)",
        "Llama 3.3 70B (Direct Groq)"
    ]
    model_choice = st.selectbox("Select AI Engine:", options=available_models, key="model_v5")
    st.divider()
    st.info("💡 Connected to Zilliz Cloud (Milvus)")

v_store, llm = init_connections(model_choice)

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

    with st.chat_message("assistant"):
        with st.spinner(f"Querying {model_choice}..."):
            # 1. Retrieval
            docs = v_store.similarity_search(prompt, k=5)
            context = "\n\n".join([f"Source: {d.metadata.get('file_name')}\n{d.page_content}" for d in docs])
            
            system_prompt = f"""
            SYSTEM: You are Freddy's Lead Recruiter. Use the following context to answer the question. 
            CONTEXT:
            {context}
            
            QUESTION: {prompt}
            """
            
            # 2. Invoke LLM
            raw_response = llm.invoke(system_prompt)
            
            # 3. 🟢 THE CLEANER (Extracting readable text only)
            if isinstance(raw_response.content, str):
                clean_text = raw_response.content
            elif isinstance(raw_response.content, list):
                clean_text = "".join([part['text'] for part in raw_response.content if 'text' in part])
            else:
                clean_text = str(raw_response.content)

            # 4. Final Output
            st.markdown(clean_text)
            st.session_state.messages.append({"role": "assistant", "content": clean_text})
