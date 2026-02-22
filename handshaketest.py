import asyncio
import nest_asyncio
import time
from datetime import datetime
nest_asyncio.apply()  # Fixes Milvus event loop crash

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus 
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# 1. INITIALIZE GLOBAL CACHE
set_llm_cache(InMemoryCache())

# 2. UI & PRICING CONFIG
st.set_page_config(page_title="Freddy's Agentic GraphRAG", layout="wide")

PRICING = {
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00}
}

if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Parallel Systems Active. Ready for Hybrid RAG."}]

# 3. HELPER FUNCTIONS
def update_usage(response, llm_object):
    """Updates the cost tracker using 2026 attribute standards."""
    model_id = getattr(llm_object, "model", "gemini-3-flash-preview")
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        in_toks = usage.get('input_tokens', usage.get('input_token_count', 0))
        out_toks = usage.get('output_tokens', usage.get('output_token_count', 0))
        rates = PRICING["gemini-2.5-pro"] if "pro" in model_id.lower() else PRICING["gemini-3-flash-preview"]
        cost = (in_toks / 1_000_000 * rates["input"]) + (out_toks / 1_000_000 * rates["output"])
        st.session_state.total_cost += cost
        st.session_state.total_tokens += (in_toks + out_toks)

def extract_clean_text(response):
    """Robust cleaner for Gemini 3 multimodal and structured response objects."""
    # 1. Handle the specific list-of-dicts format you just received
    if isinstance(response, list) and len(response) > 0:
        item = response[0]
        if isinstance(item, dict) and 'text' in item:
            return item['text']
            
    # 2. Handle LangChain Message objects (AI Message)
    if hasattr(response, 'content'):
        content = response.content
        # Sometimes content itself is a list of dicts
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        return str(content)
        
    # 3. Fallback for raw strings
    return str(response)

async def run_parallel_queries(prompt, llm, graph, v_store):
    """Executes Graph and Vector searches simultaneously to reduce latency."""
    graph_chain = GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
    
    t_start = time.time()
    # Task 1: Neo4j Cypher Execution
    task1 = asyncio.to_thread(graph_chain.invoke, {"query": prompt})
    # Task 2: Milvus Vector Search
    task2 = asyncio.to_thread(v_store.similarity_search, prompt, k=3)

    g_res, v_docs = await asyncio.gather(task1, task2)
    
    elapsed = time.time() - t_start
    v_context = "\n".join([d.page_content for d in v_docs])
    return g_res.get("result"), v_context, elapsed

# 4. CONNECTION LOGIC
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
        
        # 0.1 Temperature + Disable Thinking Budget to fix the 35s delay
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview" if "Gemini 3" in engine_choice else "gemini-2.5-pro",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.1,
            extra_body={"thinking_level": "low"}
        )

        graph = Neo4jGraph(
            url=st.secrets["NEO4J_URI"], 
            username=st.secrets["NEO4J_USERNAME"], 
            password=st.secrets["NEO4J_PASSWORD"], 
            database="73fe4e5f"
        )
        
        v_store = Milvus(
            embedding_function=embeddings, 
            collection_name="RESUME_SEARCH", 
            connection_args={"uri": st.secrets["ZILLIZ_URI"], "token": st.secrets["ZILLIZ_TOKEN"], "secure": True}
        )
        return v_store, graph, llm
    except Exception as e: return None, None, str(e)

# 5. SIDEBAR
with st.sidebar:
    st.header("💳 Session Metrics")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    st.metric("Tokens", f"{st.session_state.total_tokens:,}")
    if st.button("Reset Stats"):
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()
    st.divider()
    model_choice = st.selectbox("Engine:", ["Gemini 3 Flash", "Gemini 2.5 Pro"])
    v_store, graph, result = init_connections(model_choice)
    llm = result if v_store and graph and not isinstance(result, str) else None

# 6. MAIN CHAT LOOP
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm:
            st.error("System Offline. Check sidebar.")
        else:
            try:
                # STEP 1 & 2: Parallel Retrieval
                with st.spinner("🚀 Parallel Graph + Vector Search..."):
                    g_context, v_context, retrieval_time = asyncio.run(
                        run_parallel_queries(prompt, llm, graph, v_store)
                    )

                # STEP 3: Final Synthesis
                with st.spinner("⚖️ Final Synthesis..."):
                    t_syn_start = time.time()
                    final_prompt = f"Graph Context: {g_context}\nText Context: {v_context}\nQuestion: {prompt}"
                    ans = llm.invoke(final_prompt)
                    update_usage(ans, llm)
                    synthesis_time = time.time() - t_syn_start

                # UI Display
                full_text = extract_clean_text(ans)
                st.markdown(full_text)
                st.caption(f"⏱️ Retrieval: {retrieval_time:.2f}s | Synthesis: {synthesis_time:.2f}s")
                st.session_state.messages.append({"role": "assistant", "content": full_text})
            except Exception as e:
                st.error(f"Error: {e}")
