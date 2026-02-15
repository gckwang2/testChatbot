import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar Configuration ---
st.set_page_config(page_title="Freddy's AI Resume Search", layout="centered")

with st.sidebar:
    st.header("Model Settings")
    # Toggle Switch for Models - Excluded OpenAI
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=["Llama 3.3 70B (Free)", "Gemini 3 Flash (High Speed)"],
        index=0,
        help="Llama 3.3 is great for complex reasoning. Gemini 3 Flash is better for fast, long-context searches."
    )
    
    # OpenRouter Model Mapping for 2026
    model_map = {
        "Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
        "Gemini 3 Flash (High Speed)": "google/gemini-3-flash-preview"
    }
    
    selected_model_id = model_map[model_choice]
    st.divider()
    st.markdown(f"**Active Model:**\n`{selected_model_id}`")

# --- 2. Resource Initialization ---
@st.cache_resource
def init_resources(model_id):
    try:
        # Vector Store Connection
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
        
        # OpenRouter Configuration
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

# --- 3. Chat Application ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Hello! I am Freddy's resume assistant running on {model_choice}."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy's experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = """
        SYSTEM: You are a technical recruiter. Use the context to provide a factual answer about Freddy Goh.
        CONTEXT: {context}
        QUESTION: {question}
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner(f"Querying {model_choice}..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = chain.invoke({"query": prompt})
                ans = response["result"]
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Error: {e}")
