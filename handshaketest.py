import asyncio
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# --- 1. UI & Page Setup ---
st.set_page_config(page_title="Freddy's Agentic GraphRAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am now powered by Hybrid Graph + Vector RAG. I can see both your documents and the relationships between your skills."}
    ]

st.title("🤖 Freddy's Agentic Career Assistant")
st.caption("2026 Engine: Hybrid GraphRAG | Milvus (Vectors) + Neo4j (Relationships)")

# --- 2. THE CLEANER ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response
    if isinstance(content, list):
        return " ".join([str(i.get('text', i)) if isinstance(i, dict) else str(i) for i in content])
    return str(content)

# --- 3. Multi-Model & Multi-DB Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # --- FIX FOR GEMINI 3 CONNECTION ---
        if "Gemini 3" in engine_choice:
            # Set a thinking_budget to prevent handshake delays
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                thinking_budget=1024 # 👈 Gives the model room to reason without stalling
            )
        elif "Gemini 2.5" in engine_choice:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
        # ... other models ...

        # --- FIX FOR NEO4J DRIVER TIMEOUT ---
        graph = Neo4jGraph(
            url=st.secrets["NEO4J_URI"],
            username=st.secrets["NEO4J_USERNAME"],
            password=st.secrets["NEO4J_PASSWORD"],
            database="73fe4e5f",
            refresh_schema=False, # 👈 Set to False initially to prevent 3.0 from stalling
            driver_config={
                "connection_timeout": 60, # 👈 Increase timeout for OCI routing
                "max_connection_lifetime": 200
            }
        )
        
        # Manually refresh once after connection is stable
        graph.refresh_schema() 
        
        # ... Milvus setup remains same ...
        return v_store, graph, llm
    except Exception as e:
        return None, None, str(e)

# --- 4. Sidebar ---
with st.sidebar:
    st.header("Engine Settings")
    available_models = ["Gemini 3 Flash (Google)", "Gemini 2.5 Pro (Google)"]
    model_choice = st.selectbox("Select AI Engine:", options=available_models)
    v_store, graph, llm = init_connections(model_choice)
    
    if v_store and graph and not isinstance(llm, str):
        st.success("✅ Systems Online: Milvus + Neo4j")
    else:
        st.error(f"❌ Connection Error: {llm}")

# --- 5. Logic: Vector + Graph Search ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not v_store or not graph:
            st.error("System Offline.")
        else:
            try:
                # --- PHASE 1: Graph Retrieval (Relationships) ---
                with st.spinner("🕸️ Querying Knowledge Graph..."):
                    cypher_chain = GraphCypherQAChain.from_llm(
                        cypher_llm=llm, 
                        qa_llm=llm, 
                        graph=graph, 
                        verbose=True,
                        allow_dangerous_requests=True
                    )
                    graph_context = cypher_chain.invoke({"query": prompt})['result']

                # --- PHASE 2: Vector Retrieval (Raw Text) ---
                with st.spinner("🔍 Searching Vector Documents..."):
                    docs = v_store.similarity_search(prompt, k=3)
                    vector_context = "\n".join([d.page_content for d in docs])

                # --- PHASE 3: Hybrid Synthesis ---
                final_prompt = f"""
                You are a career advocate. Combine the structural facts from the Graph and the detailed stories from the Vector search.
                
                GRAPH FACTS: {graph_context}
                DOCUMENT DETAILS: {vector_context}
                
                QUESTION: {prompt}
                
                Synthesize a high-impact response.
                """
                
                res = llm.invoke(final_prompt)
                answer = extract_clean_text(res)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Logic Failed: {e}")
