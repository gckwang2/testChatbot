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
        {"role": "assistant", "content": "Hello! I am Freddy's Agentic Career Advocate. I don't just search; I research his 23-year career to find the best match for your needs."}
    ]

st.title("🤖 Freddy's Agentic Career Assistant")
st.caption("2026 Engine: Multi-Step Research & Synthesis (Agentic RAG)")

# --- 2. THE CLEANER: Handles Gemini/Groq dictionary outputs ---
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
        
        if "Qwen" in engine_choice:
            llm = ChatOpenAI(model="qwen3-max-2026-01-23", openai_api_key=st.secrets["QWEN_API_KEY"], openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif "Gemini" in engine_choice:
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview" if "Flash" in engine_choice else "gemini-2.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            target_model = "mixtral-8x7b-32768" if "120B" in engine_choice else "llama-3.3-70b-versatile"
            llm = ChatGroq(model=target_model, groq_api_key=st.secrets["GROQ_API_KEY"])

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
        "Gemini 3 Flash (Direct Google)", 
        "Gemini 2.5 Pro (Direct Google)", 
        "Qwen 3 Max Thinking (Alibaba)",
        "Llama 3.3 70B (Direct Groq)"
    ]
    model_choice = st.selectbox("Select AI Engine:", options=available_models)
    v_store, llm = init_connections(model_choice)
    
    if v_store:
        st.success("Connected to Zilliz")
    
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
        if not v_store:
            st.error("Database Connection Failed.")
        else:
            # --- PHASE 1: Agent Research Plan ---
            planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
            
            with st.spinner("🧠 Agent is planning research..."):
                plan_res = llm.invoke(planning_prompt)
                clean_plan = extract_clean_text(plan_res)
                search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip() and not t.startswith('-')][:3]

            # --- PHASE 2: Execution (Multi-Query Tool Use) ---
            accumulated_context = []
            # We use k=5 per query (Total 15 chunks)
            retriever = v_store.as_retriever(search_kwargs={"k": 5})
            
            for topic in search_topics:
                with st.spinner(f"🔍 Searching for: {topic}..."):
                    docs = retriever.invoke(topic)
                    accumulated_context.extend([d.page_content for d in docs])

            # --- PHASE 3: Synthesis & Advocacy ---
            # Remove duplicates from context
            context_str = "\n\n".join(list(set(accumulated_context)))
            
            final_agent_prompt = f"""
            ROLE: You are Freddy Goh's Professional Career Advocate. 
            
            CONTEXT:
            {context_str}
            
            USER QUESTION: {prompt}
            
            TASK:
            1. Analyze the context for direct evidence AND transferable skills.
            2. Since Freddy has 23+ years of experience, if a specific tool isn't listed, infer expertise based on related cloud/infrastructure seniority.
            3. Provide a professional, persuasive response focusing on business impact and leadership.
            4. Do not include JSON, metadata, or technical signatures.
            5. Do not explicitly state your role title in the response.
            """

            with st.spinner("⚖️ Synthesizing recommendation..."):
                final_res = llm.invoke(final_agent_prompt)
                answer = extract_clean_text(final_res)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
