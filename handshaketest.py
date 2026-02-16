import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

# Initialize Chat History first so the AI greets the user immediately
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's AI Career Assistant. Ask me about his expertise in LangChain, RAG, 5G, or Technical Leadership."}
    ]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Semantic Retrieval with Keyword Prioritization")

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

# --- 5. Main Interaction Logic (Semantic + Awareness) ---
if prompt := st.chat_input("Ask about Freddy's AI experience"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if v_store is None:
            st.error("Database connection failed.")
        else:
            with st.spinner("Searching records for specific skills..."):
                try:
                    # 🟢 Step 1: Semantic Search Only (No blocking filters)
                    # Increased k to 8 to provide more context for the LLM to analyze
                    docs = v_store.similarity_search(prompt, k=8)
                    
                    # 🟢 Step 2: Keyword Awareness (Detection for the LLM)
                    critical_terms = ["LangChain", "RAG", "Python", "5G", "Chatbot", "Oracle", "Zilliz", "Robotics"]
                    detected = [t for t in critical_terms if t.lower() in prompt.lower()]
                    
                    # Prepare Context and highlight keyword matches for the LLM
                    context_entries = []
                    for d in docs:
                        source = d.metadata.get('file_name', 'Experience Doc')
                        # Check if this specific snippet contains a keyword the user asked about
                        has_keyword = any(k.lower() in d.page_content.lower() for k in detected)
                        tag = " [EVIDENCE FOUND]" if has_keyword else ""
                        context_entries.append(f"SOURCE: {source}{tag}\nCONTENT: {d.page_content}")

                    context_text = "\n\n---\n\n".join(context_entries)
                    
                    # 🟢 Step 3: Prompt Engineering with Awareness
                    focus_msg = f"The user is specifically asking about: {', '.join(detected)}." if detected else ""
                    
                    system_msg = f"""
                    SYSTEM: You are Freddy's AI Career Agent. 
                    {focus_msg}
                    
                    INSTRUCTIONS:
                    1. Use the provided context to answer the question professionally.
                    2. If the user mentions keywords like {detected}, prioritize the context blocks labeled [EVIDENCE FOUND].
                    3. Do not mention that you are searching; just provide the answer based on the context.
                    4. If the information isn't in the context, mention Freddy's general AI/Oracle expertise instead.

                    CONTEXT:
                    {context_text}
                    
                    QUESTION: {prompt}
                    """

                    # 🟢 Step 4: Model Invocation and Robust Response Cleaning
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
