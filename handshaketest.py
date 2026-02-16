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
    # Standard Embeddings for Vector Search
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    # --- MODEL ROUTING LOGIC ---
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
    elif any(x in engine_choice for x in ["GPT-OSS", "Groq", "Llama"]):
        if "120B" in engine_choice:
            target_model = "mixtral-8x7b-32768" 
        else:
            target_model = "llama-3.3-70b-versatile"
            
        llm = ChatGroq(
            model=target_model, 
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )
    else:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

    # --- ZILLIZ CLOUD CONNECTION ---
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

# --- 3. Sidebar Configuration ---
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
    st.info("💡 Connected: Zilliz Cloud (Milvus)")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

v_store, llm = init_connections(model_choice)

# --- 4. Chat Interaction Loop ---
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
        with st.spinner(f"Analysing Freddy's data via {model_choice}..."):
            try:
                # 🟢 PROPERLY INDENTED BLOCK BELOW
                docs = v_store.similarity_search(prompt, k=5)
                context_blocks = []
                for d in docs:
                    fname = d.metadata.get('file_name', 'Document')
                    page = d.metadata.get('page', 'N/A')
                    context_blocks.append(f"Source: {fname} (Pg {page})\nContent: {d.page_content}")
                
                context = "\n\n---\n\n".join(context_blocks)
                
                system_prompt = f"System: Use the context to answer.\nContext: {context}\nQuestion: {prompt}"
                raw_response = llm.invoke(system_prompt)
                
                # Robust Content Extraction
                if hasattr(raw_response, 'content'):
                    content = raw_response.content
                    if isinstance(content, list):
                        clean_text = "".join([part['text'] for part in content if 'text' in part])
                    else:
                        clean_text = str(content)
                else:
                    clean_text = str(raw_response)

                st.markdown(clean_text)
                st.session_state.messages.append({"role": "assistant", "content": clean_text})
            
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
