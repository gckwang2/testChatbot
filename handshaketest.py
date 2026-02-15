import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import OracleVS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar & Model Selection ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")
st.title("🤖 Freddy's AI Career Assistant")
st.caption("AI-enabled search powered by Oracle, RAG, and 2026 flagship LLMs.")

with st.sidebar:
    st.header("Engine Settings")
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=[
            "Gemini 3 Flash (Direct Google)", 
            "Gemini 2.5 Pro (Direct Google)", 
            "Llama 3.3 70B (Direct Groq)", 
            "Qwen 3 32B (Direct Groq)",        # Updated Name
            "Llama 3.3 70B (OpenRouter Free)"
        ],
        index=0
    )
    if "Pro" in model_choice or "Qwen" in model_choice:
        st.caption("✨ Deep reasoning mode active.")

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
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
        elif engine_choice == "Gemini 2.5 Pro (Direct Google)":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                thinking_budget=1024 
            )
        elif engine_choice == "Llama 3.3 70B (Direct Groq)":
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=st.secrets["GROQ_API_KEY"]
            )
        elif engine_choice == "Qwen 3 32B (Direct Groq)":
            # UPDATED: Using the new stable 2026 ID
            llm = ChatGroq(
                model="qwen/qwen3-32b", 
                groq_api_key=st.secrets["GROQ_API_KEY"]
            )
        else:
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct:free",
                openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                openai_api_base="https://openrouter.ai/api/v1"
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
        {"role": "assistant", "content": f"Hello! Ready to chat using {model_choice}."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Retrieval Logic ---
if prompt := st.chat_input("Ask about Freddy's experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # REFINED PROMPT: Forcing the "Expert Career Coach" persona
        template = """
        You are an Expert Career Coach representing Freddy Goh. 
        Use the following pieces of context from his professional profile to answer the question.
        
        CONTEXT: {context}
        
        QUESTION: {question}
        
        If the context doesn't have the answer, use your knowledge of the tech industry 
        to explain how Freddy's stated skills would apply. Be enthusiastic!
        
        ANSWER:
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner(f"Analysing via {model_choice}..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                
                # We use chain_type_kwargs to pass the prompt into the underlying LLMChain
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = chain.invoke({"query": prompt}) # Changed to standard query dict
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Model Error: {e}")
