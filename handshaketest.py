import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus  # 🟢 Switched from OracleVS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Zilliz Cloud Hybrid Search")

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
    st.write("🌐 **Database: Zilliz Cloud**")
    st.info("Using high-precision HNSW indexing for career retrieval.")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        # 1. Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # 2. LLM Selection
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
        else:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                groq_api_key=st.secrets["GROQ_API_KEY"]
            )

        # 3. Milvus/Zilliz Vector Store
        # Using the same connection_args that worked in your ingestion script
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
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections(model_choice)

# --- 3. Retrieval Logic ---
def get_targeted_context(query, v_store):
    # Milvus handles the hybrid nature via the index. 
    # We will fetch the top 5 most relevant chunks.
    docs = v_store.similarity_search(query, k=5)
    return docs

# --- 4. Main Interaction ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about Freddy's AI projects or technical leadership."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

if prompt := st.chat_input("Does Freddy know AI Chatbots?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Zilliz Cloud..."):
            docs = get_targeted_context(prompt, v_store)
            
            # Formatting context with source info (matching our cleaned metadata)
            context_list = []
            for i, doc in enumerate(docs):
                source_file = doc.metadata.get("file_name", "Unknown")
                page_num = doc.metadata.get("page", "?")
                context_list.append(f"SOURCE: {source_file} (Pg {page_num})\nCONTENT: {doc.page_content}")
            
            full_context = "\n\n---\n\n".join(context_list)
            
            system_prompt = f"""
            SYSTEM: You are Freddy's Lead Recruiter. You must answer the question using ONLY the provided evidence.
            Be specific about his achievements (e.g., specific percentages or project names).
            
            CONTEXT:
            {full_context}
            
            QUESTION: {prompt}
            """
            
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
