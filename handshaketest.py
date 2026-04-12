import asyncio
import nest_asyncio
import time
from datetime import datetime
nest_asyncio.apply()  # Fixes Milvus event loop crash

import streamlit as st
import requests
import json
from pymilvus import connections
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain_milvus import Milvus 
# from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# 1. INITIALIZE GLOBAL CACHE
set_llm_cache(InMemoryCache())

# 2. UI & PRICING CONFIG
st.set_page_config(page_title="Freddy's Agentic GraphRAG", layout="wide")

PRICING = {
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79}
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
        if "pro" in model_id.lower():
            rates = PRICING["gemini-2.5-pro"]
        elif "llama" in model_id.lower():
            rates = PRICING["llama-3.3-70b-versatile"]
        else:
            rates = PRICING["gemini-3-flash-preview"]
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

def run_milvus_query(prompt, v_store, embeddings_model):
    """Executes Vector search using Zilliz REST API."""
    t_start = time.time()
    
    # PHASE 1: Execution (REST API Search)
    accumulated_context = []
    
    try:
        # Prepare REST details
        # Remove https:// and :443 from URI if they exist for clean formatting
        base_uri = st.secrets["ZILLIZ_URI"].replace("https://", "").replace(":443", "")
        search_url = f"https://{base_uri}/v1/vector/search"
        headers = {
            "Authorization": f"Bearer {st.secrets['ZILLIZ_TOKEN']}",
            "Content-Type": "application/json"
        }

        # Generate vector
        query_vector = embeddings_model.embed_query(prompt)
        
        # REST Payload
        payload = {
            "collectionName": "RESUME_SEARCH",
            "vector": query_vector,
            "limit": 10,
            "outputFields": ["text"]
        }
        
        # Execute HTTP POST
        response = requests.post(search_url, headers=headers, json=payload, timeout=25)
        
        if response.status_code == 200:
            results = response.json().get("data", [])
            for hit in results:
                accumulated_context.append(hit.get("text", ""))
        else:
            raise ConnectionError(f"Search Failed: {response.text}")

    except Exception as e:
        raise RuntimeError(f"REST Search Error: {e}") from e

    elapsed = time.time() - t_start
    # PHASE 2: Context Preparation
    v_context = "\n\n".join(list(set(accumulated_context)))
    return None, v_context, elapsed

# 4. CONNECTION LOGIC
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
        
        if "Gemini 3" in engine_choice:
            model_name = "gemini-3-flash-preview"
        elif "Gemini 2.5" in engine_choice:
            model_name = "gemini-2.5-pro"
        else:
            model_name = "llama-3.3-70b-versatile"

        if "Groq" in engine_choice:
            llm = ChatGroq(
                model_name=model_name,
                groq_api_key=st.secrets["GROQ_API_KEY"],
                temperature=0.1
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0.1,
                extra_body={"thinking_level": "low"}
            )

        return None, None, llm, embeddings
    except Exception as e: return None, None, str(e), None

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
    model_choice = st.selectbox("Engine:", ["Gemini 3 Flash", "Gemini 2.5 Pro", "Groq Llama 3"])
    result1, result2, result3, result4 = init_connections(model_choice)
    llm = result3 if not isinstance(result3, str) else None
    embeddings_model = result4 if not isinstance(result4, str) else None
    
    if not llm or not embeddings_model:
        st.error(f"⚠️ **System Malfunction!**")
        with st.expander("Diagnostic Report", expanded=True):
            error_message = ""
            if isinstance(result3, str):
                error_message += f"LLM Initialization Error: {result3}\n"
            if isinstance(result4, str):
                 error_message += f"Embeddings Initialization Error: {result4}\n"

            st.code(error_message if error_message else "Could not initialize LLM or Embeddings Model.", language="text")
            st.caption(f"Check st.secrets for: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `ZILLIZ_URI`, `ZILLIZ_TOKEN`.")
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
                # Retrieval spinners from working code
                search_topics = [prompt] # Use original prompt for simplicity
                for topic in search_topics:
                    with st.spinner(f"Consulting Freddy's career history..."):
                         _, v_context, retrieval_time = run_milvus_query(topic, None, embeddings_model)

                # Synthesis
                with st.spinner("⚖️ Final Synthesis..."):
                    t_syn_start = time.time()
                    final_prompt = f"Text Context: {v_context}\nQuestion: {prompt}"
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
