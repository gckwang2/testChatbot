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
    
    if "Qwen" in engine_choice:
        llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    elif "Gemini" in engine_choice:
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
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

# Sidebar is required here to avoid the "selectbox" name error
with st.sidebar:
    model_choice = st.selectbox("Model", ["Gemini 3 Flash (Direct Google)", "Llama 3.3 70B (Direct Groq)"])

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
        with st.spinner("Searching Freddy's Cloud Database..."):
            # 1. Retrieval
            docs = v_store.similarity_search(prompt, k=5)
            context = "\n\n".join([f"Source: {d.metadata.get('file_name')}\n{d.page_content}" for d in docs])
            
            system_prompt = f"Answer based on this context:\n{context}\n\nQuestion: {prompt}"
            
            # 2. Invoke LLM
            raw_response = llm.invoke(system_prompt)
            
            # 3. 🟢 THE ROBUST CLEANER: Extracting string from complex objects
            if isinstance(raw_response.content, str):
                clean_text = raw_response.content
            elif isinstance(raw_response.content, list):
                # Handles cases where content is a list of dicts like [{'type': 'text', 'text': '...'}]
                clean_text = "".join([part['text'] for part in raw_response.content if 'text' in part])
            else:
                clean_text = str(raw_response.content)

            # 4. Final Output
            st.markdown(clean_text)
            st.session_state.messages.append({"role": "assistant", "content": clean_text})
