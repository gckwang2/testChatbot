import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_oracledb import OracleVS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document  # FIXED IMPORT

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

def update_greeting():
    new_model = st.session_state.model_v4
    greeting = f"I am now using {new_model}. How can I help?"
    if "messages" in st.session_state:
        st.session_state.messages[0] = {"role": "assistant", "content": greeting}
    else:
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("2026 Engine: Python-Side Hybrid Fusion (Vector + Keyword)")

with st.sidebar:
    st.header("Engine Settings")
    available_models = [
        "Gemini 3 Flash (Direct Google)", 
        "Gemini 2.5 Pro (Direct Google)", 
        "Qwen 3 Max Thinking (Alibaba)",
        "Groq Compound (Router Model)",
        "Llama 3.3 70B (Direct Groq)"
    ]
    model_choice = st.selectbox("Select AI Engine:", options=available_models, key="model_v4", on_change=update_greeting)
    
    st.divider()
    st.write("🔧 **Hybrid Search Power**")
    keyword_boost = st.checkbox("Prioritize Exact Keyword Matches", value=True)

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
        
        if "Qwen" in engine_choice:
            llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif "Gemini" in engine_choice:
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

        v_store = OracleVS(client=conn, table_name="RESUME_SEARCH", embedding_function=embeddings)
        return conn, v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}"); st.stop()

conn, v_store, llm = init_connections(model_choice)

# --- 3. Custom Hybrid Search Logic ---
def hybrid_search_logic(query, v_store, conn, boost_keywords=True):
    # Part A: Semantic Search (Vector)
    vector_results = v_store.similarity_search(query, k=4)
    
    if not boost_keywords:
        return vector_results

    # Part B: Keyword Search (Direct SQL)
    keyword_results = []
    try:
        cursor = conn.cursor()
        search_term = f"%{query}%"
        # We search the TEXT column directly for exact matches
        cursor.execute("SELECT TEXT FROM RESUME_SEARCH WHERE TEXT LIKE :1 FETCH FIRST 2 ROWS ONLY", [search_term])
        for row in cursor:
            keyword_results.append(Document(page_content=row[0], metadata={"source": "keyword_match"}))
    except:
        pass 
    
    # Merge: Exact keyword matches are injected at the top of the list
    return keyword_results + vector_results

# --- 4. Chat Loop ---
if "messages" not in st.session_state:
    update_greeting()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Freddy's Resume..."):
            # 1. Get Hybrid Context (Python-side fusion)
            docs = hybrid_search_logic(prompt, v_store, conn, boost_keywords=keyword_boost)
            context = "\n---\n".join([d.page_content for d in docs])
            
            # 2. Generate Answer
            template = """
            SYSTEM: You are Freddy's Career Agent. Use the context below to explain why he is a fit.
            CONTEXT: {c}
            QUESTION: {q}
            ANSWER:
            """
            response = llm.invoke(template.format(c=context, q=prompt))
            
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
