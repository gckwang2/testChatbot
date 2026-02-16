import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

# --- 2. Chat History Initialization (MOVED UP) ---
# We do this before connections to ensure the "Other Party" exists immediately
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's AI Career Assistant. Ask me anything about his career journey."}
    ]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Zilliz Cloud Hybrid Search")

# --- 3. Sidebar & Engine Selection ---
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
    
    # Connection logic wrapped in a status message
    @st.cache_resource
    def init_connections(engine_choice):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
            
            if "Qwen" in engine_choice:
                llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
            elif "Gemini" in engine_choice:
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
            elif any(x in engine_choice for x in ["GPT-OSS", "Groq", "Llama"]):
                target_model = "mixtral-8x7b-32768" if "120B" in engine_choice else "llama-3.3-70b-versatile"
                llm = ChatGroq(model=target_model, groq_api_key=st.secrets["GROQ_API_KEY"])
            else:
                llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

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
        except Exception as e:
            return None, str(e)

    v_store, llm_or_error = init_connections(model_choice)
    
    if v_store is None:
        st.error(f"Offline: {llm_or_error}")
    else:
        st.success("Connected to Zilliz Cloud")

    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat reset. How can I help?"}]
        st.rerun()

# --- 4. Render Chat ---
# This loop runs every time the page refreshes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. Main Interaction ---
if prompt := st.chat_input("Ask about Freddy's AI experience"):
    # Show User message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process AI response
    if v_store is None:
        st.error("Cannot answer: Database connection is offline.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analysing evidence..."):
                try:
                    docs = v_store.similarity_search(prompt, k=5)
                    context = "\n\n".join([f"Source: {d.metadata.get('file_name')}\n{d.page_content}" for d in docs])
                    
                    system_prompt = f"System: Answer based on context.\nContext: {context}\nQuestion: {prompt}"
                    raw_response = llm_or_error.invoke(system_prompt)
                    
                    # Clean extraction
                    if hasattr(raw_response, 'content'):
                        content = raw_response.content
                        clean_text = "".join([p['text'] for p in content if 'text' in p]) if isinstance(content, list) else str(content)
                    else:
                        clean_text = str(raw_response)

                    st.markdown(clean_text)
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                except Exception as e:
                    st.error(f"Search failed: {e}")
