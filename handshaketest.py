import asyncio
import nest_asyncio
nest_asyncio.apply()  # 👈 Essential fix for Milvus + Streamlit

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_milvus import Milvus 
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# --- 1. UI & Page Setup ---
st.set_page_config(page_title="Freddy's Agentic GraphRAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's Hybrid Graph + Vector Agent. Ask me about his technical skills!"}
    ]

st.title("🤖 Freddy's Agentic Career Assistant")
st.caption("2026 Engine: Milvus (Vector) + Neo4j (Graph)")

# --- 2. Helper Logic ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response
    return str(content)

# --- 3. Robust Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    # Initialize variables as None so we don't get "not defined" errors
    v_store, graph, llm = None, None, None
    
    try:
        # A. Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # B. LLM Selection with Gemini 3 "Thinking" Fix
        if "Gemini 3" in engine_choice:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                thinking_budget=1024 # 👈 Prevents Gemini 3 from stalling the handshake
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )

        # C. Neo4j with Timeout Fix
        graph = Neo4jGraph(
            url=st.secrets["NEO4J_URI"],
            username=st.secrets["NEO4J_USERNAME"],
            password=st.secrets["NEO4J_PASSWORD"],
            database="73fe4e5f",
            refresh_schema=False, # 👈 Don't refresh immediately to avoid timeouts
            driver_config={"connection_timeout": 60}
        )
        # Safe manual refresh
        graph.refresh_schema()

        # D. Milvus
        v_store = Milvus(
            embedding_function=embeddings,
            collection_name="RESUME_SEARCH",
            connection_args={
                "uri": st.secrets["ZILLIZ_URI"],
                "token": st.secrets["ZILLIZ_TOKEN"],
                "secure": True
            }
        )
        return v_store, graph, llm

    except Exception as e:
        # If any part fails, return the error message so we can see it in UI
        return None, None, str(e)

# --- 4. Sidebar ---
with st.sidebar:
    st.header("Engine Settings")
    available_models = ["Gemini 3 Flash (Google)", "Gemini 2.5 Pro (Google)"]
    model_choice = st.selectbox("Select AI Engine:", options=available_models)
    
    # Get connections
    v_store, graph, result = init_connections(model_choice)
    
    # 'result' will contain the error string if llm is None
    if v_store and graph:
        st.success(f"✅ Connected to Milvus + Neo4j")
        llm = result # In success case, the 3rd return is the LLM
    else:
        st.error(f"❌ Connection Error: {result}")
        llm = None

# --- 5. RAG Execution ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not v_store or not graph or not llm:
            st.error("One or more systems are offline. Please check the sidebar.")
        else:
            try:
                # 1. Graph Context
                with st.spinner("🕸️ Querying Graph (Relationships)..."):
                    chain = GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
                    g_res = chain.invoke({"query": prompt})
                    graph_context = g_res.get('result', "No graph data found.")

                # 2. Vector Context
                with st.spinner("🔍 Querying Milvus (Raw Text)..."):
                    docs = v_store.similarity_search(prompt, k=3)
                    vector_context = "\n".join([d.page_content for d in docs])

                # 3. Final Answer
                final_prompt = f"Combine these facts into a career advocacy response:\nGraph: {graph_context}\nText: {vector_context}\nQuestion: {prompt}"
                ans = llm.invoke(final_prompt)
                st.markdown(extract_clean_text(ans))
                st.session_state.messages.append({"role": "assistant", "content": extract_clean_text(ans)})
            except Exception as e:
                st.error(f"Query failed: {e}")
