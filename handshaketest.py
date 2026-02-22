import asyncio
import nest_asyncio
import time  # 👈 Added for high-precision timing
from datetime import datetime
nest_asyncio.apply()

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus 
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Activate Semantic Cache
set_llm_cache(InMemoryCache())

# --- 1. UI & Pricing Setup ---
st.set_page_config(page_title="Freddy's Agentic GraphRAG", layout="wide")

PRICING = {
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00}
}

if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Systems Online. Performance tracking active."}]

# --- 2. Timing Utility ---
def log_stage(stage_name, start_time):
    """Calculates elapsed time and prints a formatted log."""
    elapsed = time.time() - start_time
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] ⏱️ {stage_name}: {elapsed:.2f}s"
    print(log_msg)  # Prints to your terminal/OCI logs
    return log_msg, time.time() # Returns message and new reference start time

# --- [Keep Helper Functions extract_clean_text and update_usage from before] ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        return str(content)
    return str(response)

def update_usage(response, llm_object):
    model_id = getattr(llm_object, "model", "gemini-3-flash-preview")
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        in_toks = usage.get('input_tokens', usage.get('input_token_count', 0))
        out_toks = usage.get('output_tokens', usage.get('output_token_count', 0))
        rates = PRICING["gemini-2.5-pro"] if "pro" in model_id.lower() else PRICING["gemini-3-flash-preview"]
        cost = (in_toks / 1_000_000 * rates["input"]) + (out_toks / 1_000_000 * rates["output"])
        st.session_state.total_cost += cost
        st.session_state.total_tokens += (in_toks + out_toks)

# --- 3. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview" if "Gemini 3" in engine_choice else "gemini-2.5-pro",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        graph = Neo4jGraph(url=st.secrets["NEO4J_URI"], username=st.secrets["NEO4J_USERNAME"], password=st.secrets["NEO4J_PASSWORD"], database="73fe4e5f")
        v_store = Milvus(embedding_function=embeddings, collection_name="RESUME_SEARCH", 
                        connection_args={"uri": st.secrets["ZILLIZ_URI"], "token": st.secrets["ZILLIZ_TOKEN"], "secure": True})
        return v_store, graph, llm
    except Exception as e: return None, None, str(e)

# --- 4. Sidebar ---
with st.sidebar:
    st.header("💳 Usage & Performance")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    st.metric("Tokens", f"{st.session_state.total_tokens:,}")
    model_choice = st.selectbox("Engine:", ["Gemini 3 Flash (Google)", "Gemini 2.5 Pro (Google)"])
    v_store, graph, result = init_connections(model_choice)
    llm = result if v_store and graph and not isinstance(result, str) else None

# --- 5. Hybrid RAG Logic with Timing ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy's career..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            query_start = time.time()
            perf_logs = []

            # Stage 1: Graph Search
            with st.spinner("🕸️ Graph Search..."):
                t_ref = time.time()
                graph_chain = GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
                g_res = graph_chain.invoke({"query": prompt})
                log, t_ref = log_stage("Graph RAG", t_ref)
                perf_logs.append(log)

            # Stage 2: Vector Search
            with st.spinner("🔍 Vector Search..."):
                docs = v_store.similarity_search(prompt, k=3)
                v_context = "\n".join([d.page_content for d in docs])
                log, t_ref = log_stage("Vector Retrieval", t_ref)
                perf_logs.append(log)

            # Stage 3: Synthesis
            with st.spinner("⚖️ Final Synthesis..."):
                ans = llm.invoke(f"Graph: {g_res['result']}\nText: {v_context}\nQ: {prompt}")
                update_usage(ans, llm)
                log, t_ref = log_stage("LLM Synthesis", t_ref)
                perf_logs.append(log)

            # Final Display
            total_time = time.time() - query_start
            full_text = extract_clean_text(ans)
            st.markdown(full_text)
            
            # Show performance footer
            st.caption(f"⏱️ Total: {total_time:.2f}s | " + " | ".join(perf_logs))
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            
        except Exception as e:
            st.error(f"Error: {e}")
