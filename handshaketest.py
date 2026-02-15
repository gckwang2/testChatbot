import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_openai import ChatOpenAI

# 2026 FIX: Import from langchain_classic instead of langchain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. App Configuration ---
st.set_page_config(page_title="Freddy's AI Resume Search", layout="centered")

with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=["Llama 3.3 70B (Free)", "Gemini 3 Flash (High Speed)"],
        index=0
    )
    
    model_map = {
        "Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
        "Gemini 3 Flash (High Speed)": "google/gemini-3-flash-preview"
    }
    selected_model_id = model_map[model_choice]

# --- 2. Resource Initialization ---
@st.cache_resource
def init_resources(model_id):
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
        
        v_store = OracleVS(
            client=conn, 
            table_name="RESUME_SEARCH", 
            embedding_function=embeddings
        )
        
        llm = ChatOpenAI(
            model=model_id,
            openai_api_key=st.secrets["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        return v_store, llm
    except Exception as e:
        st.error(f"Setup Failed: {e}")
        st.stop()

v_store, llm = init_resources(selected_model_id)

# --- 3. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Ready! Using {model_choice}."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. RAG Logic ---
if prompt := st.chat_input("Ask about Freddy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        system_prompt = (
            "You are a professional recruiter. Use the context to answer about Freddy Goh.\n\n"
            "Context: {context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        with st.spinner(f"Querying {model_choice}..."):
            try:
                combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                # Using the classic retrieval chain bridge
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                
                response = retrieval_chain.invoke({"input": prompt})
                ans = response["answer"]
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Error: {e}")
