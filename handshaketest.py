import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

# Initialize Chat History first to ensure "Other Party" renders immediately
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's AI Career Assistant. Ask me about his expertise in LangChain, RAG, 5G, or Technical Leadership."}
    ]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Zilliz Cloud Hybrid Search (Keyword + Semantic)")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Model Selection
        if "Qwen" in engine_choice:
            llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif "Gemini" in engine_choice:
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            target_model = "mixtral-8x7b-32768" if "120B" in engine_choice else "llama-3.3-70b-versatile"
            llm = ChatGroq(model=target_model, groq_api_key=st.secrets["GROQ_API_KEY"])

        # Milvus/Zilliz Store
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
    
    v_store, llm_or_err = init_connections(model_choice)
    
    if v_store is None:
        st.error(f"Offline: {llm_or_err}")
    else:
        st.success("Connected: Zilliz Cloud")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat reset. How can I help?"}]
        st.rerun()

# --- 4. Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. Main Interaction Logic ---
if prompt := st.chat_input("Ask about Freddy's AI experience"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if v_store is None:
            st.error("Database connection failed. Check secrets.")
        else:
            with st.spinner("Hybrid Search: Filtering for keywords & intent..."):
                try:
                    # 🟢 KEYWORD DETECTION OPTIMIZATION
                    # Define words that MUST trigger exact matches
                    critical_terms = ["LangChain", "RAG", "Python", "5G", "Chatbot", "Oracle", "Zilliz"]
                    detected = [t for t in critical_terms if t.lower() in prompt.lower()]
                    
                    filter_expr = None
                    if detected:
                        # Tells Milvus to look specifically for these keywords in the text field
                        filter_expr = " or ".join([f"text LIKE '%{t}%'" for t in detected])

                    # Step 1: Attempt Hybrid Search
                    docs = v_store.similarity_search(prompt, k=5, expr=filter_expr)
                    
                    # Step 2: Fallback if keyword filter was too restrictive
                    if not docs:
                        docs = v_store.similarity_search(prompt, k=5)

                    # Prepare Context
                    context_text = "\n\n".join([f"Source: {d.metadata.get('file_name', 'Doc')}\n{d.page_content}" for d in docs])
                    
                    # LLM Prompt
                    system_msg = f"""
                    SYSTEM: You are Freddy's Career Advocate. Answer based ONLY on the evidence.
                    KEYWORDS DETECTED: {', '.join(detected) if detected else 'None'}
                    
                    CONTEXT FROM DATABASE:
                    {context_text}
                    
                    QUESTION: {prompt}
                    """

                    # Invoke & Robust Clean
                    raw_res = llm_or_err.invoke(system_msg)
                    if hasattr(raw_res, 'content'):
                        txt = raw_res.content
                        clean_text = "".join([p['text'] for p in txt if 'text' in p]) if isinstance(txt, list) else str(txt)
                    else:
                        clean_text = str(raw_res)

                    st.markdown(clean_text)
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})

                except Exception as e:
                    st.error(f"Search failed: {e}")
