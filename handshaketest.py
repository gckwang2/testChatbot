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
        # A. Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # B. LLM Selection
        if "Gemini" in engine_choice:
            target = "gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro"
            llm = ChatGoogleGenerativeAI(model=target, google_api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            # Simplified fallback for this example
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

        # C. Milvus (Vector Store)
        v_store = Milvus(
            embedding_function=embeddings,
            collection_name="RESUME_SEARCH",
            connection_args={
                "uri": st.secrets["ZILLIZ_URI"],
                "token": st.secrets["ZILLIZ_TOKEN"],
                "secure": True
            }
        )

        # D. Neo4j (Graph Store) - Using your confirmed OCI details
        graph = Neo4jGraph(
            url=st.secrets["NEO4J_URI"],
            username=st.secrets["NEO4J_USERNAME"],
            password=st.secrets["NEO4J_PASSWORD"],
            database="73fe4e5f",
            refresh_schema=True # Important to see your 959 nodes!
        )
        
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
