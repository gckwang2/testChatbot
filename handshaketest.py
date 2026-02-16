import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_oracledb import OracleVS, OracleHybridSearchRetriever
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Sidebar & Callback Logic ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

def update_greeting():
    new_model = st.session_state.model_selector_v3
    greeting = f"I am now using {new_model}. How can I help?"
    if "messages" in st.session_state:
        st.session_state.messages[0] = {"role": "assistant", "content": greeting}
    else:
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Oracle 23ai Hybrid Search (Protocol Fix)")

with st.sidebar:
    st.header("Engine Settings")
    available_models = [
        "Gemini 3 Flash (Direct Google)", 
        "Gemini 2.5 Pro (Direct Google)", 
        "Qwen 3 Max Thinking (Alibaba)",
        "Groq Compound (Router Model)",
        "GPT-OSS-120B (Direct Groq)",
        "Llama 3.3 70B (Direct Groq)", 
        "Llama 3.3 70B (OpenRouter Free)"
    ]
    
    model_choice = st.selectbox(
        "Select AI Engine:",
        options=available_models,
        index=0,
        key="model_selector_v3",
        on_change=update_greeting
    )
    
    st.divider()
    st.write("🔧 **Hybrid Weights**")
    # Oracle weights usually perform best between 0 and 1
    t_weight = st.slider("Keyword (Text) Weight", 0.0, 1.0, 0.5, 0.05)
    v_weight = st.slider("Semantic (Vector) Weight", 0.0, 1.0, 0.5, 0.05)

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
        
        # LLM Logic
        if engine_choice == "Gemini 3 Flash (Direct Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=st.secrets["GOOGLE_API_KEY"])
        elif engine_choice == "Gemini 2.5 Pro (Direct Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"], thinking_budget=1024)
        elif engine_choice == "Qwen 3 Max Thinking (Alibaba)":
            llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif engine_choice == "Groq Compound (Router Model)":
            llm = ChatGroq(model="groq/compound", groq_api_key=st.secrets["GROQ_API_KEY"])
        else:
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

        v_store = OracleVS(client=conn, table_name="RESUME_SEARCH", embedding_function=embeddings)
        
        HYBRID_INDEX_NAME = "hybrid_idx_resume"
        return conn, v_store, llm, HYBRID_INDEX_NAME
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

conn, v_store, llm, HYBRID_INDEX_NAME = init_connections(model_choice)

# --- 3. Chat Session State ---
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
        template = """
        SYSTEM: You are Freddy's Career Assistant.
        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Executing Hybrid Search..."):
            try:
                # FIXED: Structural change to params to match Oracle 23ai/26ai requirements
                # The 'text' and 'vector' keys are mandatory sub-objects for weights.
                retriever = OracleHybridSearchRetriever(
                    client=conn,
                    vector_store=v_store,
                    idx_name=HYBRID_INDEX_NAME,
                    search_mode="hybrid",
                    k=5,
                    params={
                        "text": {"score_weight": t_weight},
                        "vector": {"score_weight": v_weight}
                    }
                )

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
                st.error(f"Hybrid Engine Error: {e}")
                st.info("Attempting automatic fallback to standard Vector Search...")
                
                # Dynamic Fallback: Ensures the user always gets an answer
                fallback_retriever = v_store.as_retriever(search_kwargs={"k": 5})
                fallback_chain = RetrievalQA.from_chain_type(llm=llm, retriever=fallback_retriever)
                res = fallback_chain.invoke({"query": prompt})
                st.markdown(res["result"])
