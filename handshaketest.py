import asyncio
import nest_asyncio
nest_asyncio.apply()  # 👈 MUST BE AT THE TOP: Fixes the Milvus event loop crash

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_milvus import Milvus 
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# --- 1. UI & Page Setup ---
st.set_page_config(page_title="Freddy's Hybrid Agent", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System initialized. Vector (Milvus) and Graph (Neo4j) modules are ready."}
    ]

st.title("🤖 Freddy's Agentic Career Assistant")
st.caption("2026 Engine: Hybrid GraphRAG | High-Recall Search")

# --- 2. THE CLEANER (Updated for Gemini 3 Multimodal Blocks) ---
def extract_clean_text(response):
    """Digs into the complex response objects returned by Gemini 3.0."""
    # Handle the specific [{'type': 'text', 'text': '...'}] format
    if isinstance(response, list) and len(response) > 0:
        item = response[0]
        if isinstance(item, dict) and 'text' in item:
            return item['text']
    
    # Handle LangChain Message objects
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        return str(content)
    
    return str(response)

# --- 3. Multi-DB Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    v_store, graph, llm = None, None, None
    try:
        # A. Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # B. LLM Selection (Gemini 3 Flash 2026 Config)
        if "Gemini 3" in engine_choice:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                thinking_level="low", # 👈 Faster for RAG handshake than "high"
                temperature=1.0      # 👈 Gemini 3 standard for reasoning
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )

        # C. Neo4j Connection (with OCI Timeout Protection)
        graph = Neo4jGraph(
            url=st.secrets["NEO4J_URI"],
            username=st.secrets["NEO4J_USERNAME"],
            password=st.secrets["NEO4J_PASSWORD"],
            database="73fe4e5f",
            refresh_schema=False, # 👈 Manual refresh below to avoid startup hang
            driver_config={"connection_timeout": 60}
        )
        graph.refresh_schema()

        # D. Milvus Connection
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

# --- 4. Sidebar Engine Selection ---
with st.sidebar:
    st.header("System Control")
    available_models = ["Gemini 3 Flash (Google)", "Gemini 2.5 Pro (Google)"]
    model_choice = st.selectbox("Select Intelligence Engine:", options=available_models)
    
    v_store, graph, result = init_connections(model_choice)
    
    if v_store and graph and not isinstance(result, str):
        st.success("🟢 All Systems Online")
        llm = result
    else:
        st.error(f"🔴 System Error: {result}")
        llm = None

# --- 5. The Hybrid RAG Logic ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not v_store or not graph or not llm:
            st.error("System offline. Please check API keys and Sidebar.")
        else:
            try:
                # PHASE 1: Graph Relationship Search
                with st.spinner("🕸️ Querying Skills Graph..."):
                    # allow_dangerous_requests is required for write-capable drivers in 2026
                    graph_chain = GraphCypherQAChain.from_llm(
                        llm, 
                        graph=graph, 
                        allow_dangerous_requests=True,
                        verbose=True
                    )
                    graph_data = graph_chain.invoke({"query": prompt})['result']

                # PHASE 2: Vector Document Search
                with st.spinner("🔍 Retrieving Document Context..."):
                    docs = v_store.similarity_search(prompt, k=3)
                    vector_data = "\n\n".join([d.page_content for d in docs])

                # PHASE 3: Synthesis
                synthesis_prompt = f"""
                As Freddy's Career Advocate, synthesize an answer using these two sources:
                1. RELATIONSHIP DATA (GRAPH): {graph_data}
                2. TEXTUAL CONTEXT (VECTOR): {vector_data}
                
                USER QUESTION: {prompt}
                
                Provide a professional, formatted response. Focus on Freddy's 23+ years of impact.
                """
                
                with st.spinner("⚖️ Synthesizing Hybrid Answer..."):
                    final_res = llm.invoke(synthesis_prompt)
                    answer = extract_clean_text(final_res)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
            except Exception as e:
                st.error(f"Agentic Cycle Failed: {e}")
