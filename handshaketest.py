import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import OracleVS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar & Model Selection ---
st.set_page_config(page_title="Freddy's Resume AI", layout="centered")

with st.sidebar:
    st.header("Engine Settings")
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=[
            "Gemini 3 Flash (Direct Google)", 
            "Llama 3.3 70B (Direct Groq)", 
            "Llama 3.3 70B (OpenRouter Free)"
        ],
        index=0
    )
    st.info(f"Currently active: {model_choice}")

# --- 2. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        # DB Connection
        conn = oracledb.connect(
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            dsn=st.secrets["DB_DSN"]
        )
        
        # Embeddings (Always using your Google Key)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # LLM Logic for the 3 Options
        if engine_choice == "Gemini 3 Flash (Direct Google)":
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
        elif engine_choice == "Llama 3.3 70B (Direct Groq)":
            # Direct Groq Connection - Fastest Option
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=st.secrets["GROQ_API_KEY"]
            )
        else:
            # OpenRouter Fallback
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct:free",
                openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=1000
            )

        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH", 
            embedding_function=embeddings
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections(model_choice)

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I'm using {model_choice}. How can I help?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Retrieval Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = """
        SYSTEM: You are an expert Career Coach. Use the context to answer about Freddy Goh.
        
        CONTEXT: {context}
        QUESTION: {question}
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner(f"Searching via {model_choice}..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = chain.invoke(prompt)
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Search Error: {e}")
