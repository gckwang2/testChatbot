import asyncio
import nest_asyncio
nest_asyncio.apply()  # 👈 MUST BE AT THE TOP: Fixes the Milvus event loop crash

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_milvus import Milvus 
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
# Add this to the top of your script or inside init_connections
set_llm_cache(InMemoryCache())

# --- 1. Pricing Configuration (Gemini 2026 Rates) ---
# Rates per 1M tokens (approximate)
PRICING = {
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00}
}

# Initialize session state for costs
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

def update_usage(response, model_name):
    """Parses usage_metadata and updates global costs."""
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        in_toks = usage.get('input_token_count', 0)
        out_toks = usage.get('output_token_count', 0)
        
        # Calculate cost
        rates = PRICING.get(model_name, PRICING["gemini-3-flash-preview"])
        cost = (in_toks / 1_000_000 * rates["input"]) + (out_toks / 1_000_000 * rates["output"])
        
        st.session_state.total_cost += cost
        st.session_state.total_tokens += (in_toks + out_toks)

# --- [Keep sections 2-4 from previous script, but update the sidebar] ---

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

# --- 5. The Hybrid RAG Logic (Updated for Tracking) ---
if prompt := st.chat_input("Ask about Freddy..."):
    # ... previous setup code ...
    
    with st.chat_message("assistant"):
        try:
            # 1. Graph Context
            with st.spinner("🕸️ Querying Graph..."):
                # Use base LLM to get the usage later
                graph_chain = GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
                # Note: CypherQAChain wraps several calls, manual tracking below is better for accuracy
                g_res = graph_chain.invoke({"query": prompt})
                # If your chain doesn't expose usage directly, the LLM synthesis will capture it
            
            # 2. Vector Context (Milvus)
            with st.spinner("🔍 Querying Milvus..."):
                docs = v_store.similarity_search(prompt, k=3)
                v_context = "\n".join([d.page_content for d in docs])

            # 3. Final Synthesis & Tracking
            with st.spinner("⚖️ Synthesizing..."):
                final_prompt = f"Context:\nGraph: {g_res['result']}\nText: {v_context}\nQuestion: {prompt}"
                ans = llm.invoke(final_prompt)
                
                # UPDATE COST TRACKER HERE
                update_usage(ans, llm.model_name)
                
                st.markdown(extract_clean_text(ans))
                st.session_state.messages.append({"role": "assistant", "content": extract_clean_text(ans)})
        except Exception as e:
            st.error(f"Error: {e}"
