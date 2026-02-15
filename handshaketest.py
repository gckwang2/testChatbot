import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import OracleVS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar & Callback Logic ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

# This function updates the greeting message when you switch models
def update_greeting():
    new_model = st.session_state.model_selector
    greeting = f"I am now using {new_model}. How can I help?"
    if "messages" in st.session_state:
        # Update the very first assistant message
        st.session_state.messages[0] = {"role": "assistant", "content": greeting}
    else:
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 flagship search: Oracle Vector + RAG + Multi-LLM Selection")

with st.sidebar:
    st.header("Engine Settings")
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=[
            "Gemini 3 Flash (Direct Google)", 
            "Gemini 2.5 Pro (Direct Google)", 
            "Llama 3.3 70B (Direct Groq)", 
            "Qwen 3 32B (Direct Groq)", 
            "Llama 3.3 70B (OpenRouter Free)"
        ],
        index=0,
        key="model_selector",      # Added key for session state access
        on_change=update_greeting # This triggers the update immediately
    )
    if "Pro" in model_choice or "Qwen" in model_choice:
        st.caption("✨ Reasoning Mode: Enabled")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        conn = oracledb.connect(
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            dsn=st.secrets["DB_DSN"]
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        if engine_choice == "Gemini 3 Flash (Direct Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=st.secrets["GOOGLE_API_KEY"])
        elif engine_choice == "Gemini 2.5 Pro (Direct Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"], thinking_budget=1024)
        elif engine_choice == "Llama 3.3 70B (Direct Groq)":
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])
        elif engine_choice == "Qwen 3 32B (Direct Groq)":
            llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=st.secrets["GROQ_API_KEY"])
        else:
            llm = ChatOpenAI(model="meta-llama/llama-3.3-70b-instruct:free", openai_api_key=st.secrets["OPENROUTER_API_KEY"], openai_api_base="https://openrouter.ai/api/v1")

        v_store = OracleVS(client=conn, table_name="RESUME_SEARCH", embedding_function=embeddings)
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections(model_choice)

# --- 3. Chat Session State ---
# Ensure initialization happens if it hasn't yet
if "messages" not in st.session_state:
    update_greeting()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Retrieval & Prompt Loop ---
if prompt := st.chat_input("Ask about Freddy's experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = f"""
        SYSTEM: You are an Expert Career Coach representing Freddy Goh. 
        You are currently operating via the {model_choice} engine.
        
        CONTEXT: {{context}}
        QUESTION: {{question}}
        
        INSTRUCTIONS: Use the context above to highlight Freddy's achievements. 
        ANSWER:
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner(f"Processing with {model_choice}..."):
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
