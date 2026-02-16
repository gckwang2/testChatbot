import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage

# --- 1. UI & Page Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's Agentic Career Advocate. I'm ready to research his 23-year career across all technical domains."}
    ]

st.title("🤖 Freddy's Agentic Career Assistant")
st.caption("2026 Engine: Agentic RAG | High-Recall Multi-Query Search")

# --- 2. THE CLEANER ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response

    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content[0]))
        return " ".join([str(i) for i in content])
    
    return str(content)

# --- 3. Multi-Model Connection Logic ---
@st.cache_resource
def init_connections(engine_choice):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # 2026 Optimized Model IDs
        if "Qwen" in engine_choice:
            llm = ChatOpenAI(
                model="qwen3-max-2026-01-23", 
                openai_api_key=st.secrets["QWEN_API_KEY"], 
                openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        elif "Gemini" in engine_choice:
            target = "gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro"
            llm = ChatGoogleGenerativeAI(model=target, google_api_key=st.secrets["GOOGLE_API_KEY"])
        elif "GPT-OSS-120B" in engine_choice:
            llm = ChatGroq(model="gpt-oss-120b", groq_api_key=st.secrets["GROQ_API_KEY"])
        elif "Llama 4 Scout" in engine_choice:
            llm = ChatGroq(model="llama-4-scout-17b", groq_api_key=st.secrets["GROQ_API_KEY"])
        else:
            # Fallback for Groq Compound/Others
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

        v_store = Milvus(
            embedding_function=embeddings,
            collection_name="RESUME_SEARCH",
            connection_args={
                "uri": st.secrets["ZILLIZ_URI"],
                "token": st.secrets["ZILLIZ_TOKEN"],
                "secure": True
            }
        )
        return v_store, llm
    except Exception as e:
        return None, str(e)

# --- 4. Sidebar Engine Selection ---
with st.sidebar:
    st.header("Engine Settings")
    available_models = [
        "Groq Compound (Router)",
        "GPT-OSS-120B (Groq)",
        "Llama 4 Scout 17B 16E (Groq)",
        "Gemini 3 Flash (Google)", 
        "Gemini 2.5 Pro (Google)", 
        "Qwen 3 Max Thinking (Alibaba)"
    ]
    model_choice = st.selectbox("Select AI Engine:", options=available_models)
    v_store, llm = init_connections(model_choice)
    
    if v_store and not isinstance(llm, str):
        st.success(f"Online: {model_choice}")
    elif isinstance(llm, str):
        st.error(f"Connection Error: {llm}")
    
    if st.button("Clear History"):
        st.session_state.messages = [{"role": "assistant", "content": "Research reset. How can I help?"}]
        st.rerun()

# --- 5. Display History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. The Agentic Logic ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not v_store or isinstance(llm, str):
            st.error("System is offline. Check API keys and Model selection.")
        else:
            try:
                # --- PHASE 1: Agent Research Plan ---
                planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
                
                with st.spinner("🧠 Agent Planning Research..."):
                    plan_res = llm.invoke(planning_prompt)
                    clean_plan = extract_clean_text(plan_res)
                    search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip() and not t.startswith('-')][:3]

                # --- PHASE 2: Execution ---
                accumulated_context = []
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                
                for topic in search_topics:
                    with st.spinner(f"🔍 Searching: {topic}..."):
                        docs = retriever.invoke(topic)
                        accumulated_context.extend([d.page_content for d in docs])

                # --- PHASE 3: Synthesis & Advocacy ---
                context_str = "\n\n".join(list(set(accumulated_context)))
                
                final_agent_prompt = f"""
                ROLE: Professional Career Advocate. 
                
                CONTEXT:
                {context_str}
                
                USER QUESTION: {prompt}
                
                TASK:
                1. Analyze context for direct evidence and transferable skills.
                2. Given Freddy's 23+ years of seniority, infer expertise for related technologies (e.g., if Cloud Architecture is found, infer platform adaptability).
                3. Focus on leadership and high-level business impact.
                4. No JSON or technical signatures. Do not mention your role title.
                """

                with st.spinner("⚖️ Synthesizing Final Answer..."):
                    final_res = llm.invoke(final_agent_prompt)
                    answer = extract_clean_text(final_res)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Agent Logic Failed: {e}. Try a different model in the sidebar.")
