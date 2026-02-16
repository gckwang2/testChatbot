import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_oracledb import OracleVS 
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar & Callback Logic ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

def update_greeting():
    # Note: Using the new key here as well
    new_model = st.session_state.model_selector_v2
    greeting = f"I am now using {new_model}. How can I help?"
    if "messages" in st.session_state:
        st.session_state.messages[0] = {"role": "assistant", "content": greeting}
    else:
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Oracle 23ai + Qwen 3 Max Flagship")

with st.sidebar:
    st.header("Engine Settings")
    
    # Force the list to be fresh
    available_models = [
        "Gemini 3 Flash (Direct Google)", 
        "Gemini 2.5 Pro (Direct Google)", 
        "Qwen 3 Max (Direct Alibaba)",
        "Qwen 3 Max Thinking (Alibaba)",
        "Groq Compound (Router Model)",
        "GPT-OSS-120B (Direct Groq)",
        "Llama 3.3 70B (Direct Groq)", 
        "Llama 3.3 70B (OpenRouter Free)"
    ]
    
    # Using 'model_selector_v2' to force a widget refresh
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=available_models,
        index=0,
        key="model_selector_v2",
        on_change=update_greeting
    )
    
    if "Max" in model_choice:
        st.success("🔥 Qwen 3 Max Flagship Active")

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
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # LLM Mapping
        if engine_choice == "Gemini 3 Flash (Direct Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=st.secrets["GOOGLE_API_KEY"])
        
        elif engine_choice == "Gemini 2.5 Pro (Direct Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"], thinking_budget=1024)
        
        elif engine_choice == "Qwen 3 Max (Direct Alibaba)":
            llm = ChatOpenAI(
                model="qwen3-max", 
                openai_api_key=st.secrets["QWEN_API_KEY"],
                openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

        elif engine_choice == "Qwen 3 Max Thinking (Alibaba)":
            llm = ChatOpenAI(
                model="qwen3-max-2026-01-23", 
                openai_api_key=st.secrets["QWEN_API_KEY"],
                openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
        elif engine_choice == "Groq Compound (Router Model)":
            llm = ChatGroq(model="groq/compound", groq_api_key=st.secrets["GROQ_API_KEY"])
            
        elif engine_choice == "GPT-OSS-120B (Direct Groq)":
            llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=st.secrets["GROQ_API_KEY"])
            
        elif engine_choice == "Llama 3.3 70B (Direct Groq)":
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])
            
        else:
            llm = ChatOpenAI(model="meta-llama/llama-3.3-70b-instruct:free", openai_api_key=st.secrets["OPENROUTER_API_KEY"], openai_api_base="https://openrouter.ai/api/v1")

        v_store = OracleVS(client=conn, table_name="RESUME_SEARCH", embedding_function=embeddings)
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections(model_choice)

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    update_greeting()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. RAG Retrieval Loop ---
if prompt := st.chat_input("Ask about Freddy's experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = """
        SYSTEM: You are a Career Coach for Freddy Goh.
        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner(f"Running {model_choice}..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Model Error: {e}")
