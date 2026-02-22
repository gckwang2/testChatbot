import asyncio
import nest_asyncio
nest_asyncio.apply()  # 👈 Fixes the Milvus event loop crash

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_milvus import Milvus 
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Activate Semantic Cache
set_llm_cache(InMemoryCache())

# --- 1. UI & Pricing Configuration ---
st.set_page_config(page_title="Freddy's Agentic GraphRAG", layout="wide")

PRICING = {
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00}
}

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hybrid RAG Systems Online. How can I assist with Freddy's career data?"}]

# --- 2. Helper Functions ---
def extract_clean_text(response):
    if isinstance(response, list) and len(response) > 0:
        item = response[0]
        if isinstance(item, dict) and 'text' in item:
            return item['text']
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        return str(content)
    return str(response)

def update_usage(response, llm_object):
    """Parses usage_metadata and updates global costs using the 2026 attribute schema."""
    # LangChain 4.0 uses .model instead of .model_name
    model_id = getattr(llm_object, "model", "gemini-3-flash-preview")
    
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        # 2026 standardized keys: input_tokens / output_tokens
        in_toks = usage.get('input_tokens', usage.get('input_token_count', 0))
        out_toks = usage.get('output_tokens', usage.get('output_token_count', 0))
        
        # Select rates
        if "pro" in model_id.lower():
            rates = PRICING["gemini-2.5-pro"]
        else:
            rates = PRICING["gemini-3-flash-preview"]
            
        cost = (in_toks / 1_000_000 * rates["input"]) + (out_toks / 1_000_000 * rates["output"])
        
        st.session_state.total_cost += cost
        st.session_state.total_tokens += (in_toks + out_toks)

# --- 3. Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        if "Gemini 3" in engine_choice:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                thinking_budget=1024
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )

        graph = Neo4jGraph(
            url=st.secrets["NEO4J_URI"],
            username=st.secrets["NEO4J_USERNAME"],
            password=st.secrets["NEO4J_PASSWORD"],
            database="73fe4e5f",
            refresh_schema=False,
            driver_config={"connection_timeout": 60}
        )
        graph.refresh_schema()

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
        return None, None, str(e)

# --- 4. Sidebar ---
with st.sidebar:
    st.header("💳 Usage & Cost Tracker")
    col1, col2 = st.columns(2)
    col1.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    col2.metric("Tokens Used", f"{st.session_state.total_tokens:,}")
    
    if st.button("Reset Billing"):
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()
    
    st.divider()
    model_choice = st.selectbox("Intelligence Engine:", ["Gemini 3 Flash (Google)", "Gemini 2.5 Pro (Google)"])
    v_store, graph, result = init_connections(model_choice)
    
    if v_store and graph and not isinstance(result, str):
        st.success("🟢 Systems Online")
        llm = result
    else:
        st.error(f"🔴 Offline: {result}")

# --- 5. Hybrid RAG Logic ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not v_store or not graph or not llm:
            st.error("Connection lost. Check sidebar.")
        else:
            try:
                # 1. Graph Context
                with st.spinner("🕸️ Querying Graph..."):
                    graph_chain = GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
                    g_res = graph_chain.invoke({"query": prompt})
                
                # 2. Vector Context
                with st.spinner("🔍 Querying Milvus..."):
                    docs = v_store.similarity_search(prompt, k=3)
                    v_context = "\n".join([d.page_content for d in docs])

                # 3. Final Synthesis
                with st.spinner("⚖️ Synthesizing..."):
                    final_prompt = f"Graph Context: {g_res['result']}\nText Context: {v_context}\nQuestion: {prompt}"
                    ans = llm.invoke(final_prompt)
                    update_usage(ans, llm)
                    
                    full_text = extract_clean_text(ans)
                    st.markdown(full_text)
                    st.session_state.messages.append({"role": "assistant", "content": full_text})
            except Exception as e:
                st.error(f"Error: {e}") # 👈 Bracket closed!
