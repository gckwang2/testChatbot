import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.title("🤖 Freddy's AI Career Assistant")
st.caption("Enhanced Semantic RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. Connections ---
@st.cache_resource
def init_connections():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        v_store = Milvus(
            embedding_function=embeddings,
            connection_args={
                "uri": st.secrets["ZILLIZ_URI"],
                "token": st.secrets["ZILLIZ_TOKEN"],
                "secure": True
            },
            collection_name="RESUME_SEARCH"
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections()

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Freddy's AI career assistant. How can I help you explore his 23 years of expertise today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Optimized Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 🟢 KEYWORD DETECTION (Non-filtering)
        # We detect these to tell the LLM to prioritize them, but we DON'T filter the database with them.
        critical_keywords = ["LangChain", "RAG", "Python", "5G", "Robotics", "Oracle", "Zilliz", "Chatbot"]
        detected = [k for k in critical_keywords if k.lower() in prompt.lower()]
        
        # Build the dynamic instruction
        detection_hint = f"Note: The user is specifically asking about {', '.join(detected)}." if detected else ""

        template = f"""
        SYSTEM: You are a professional recruiter and Freddy's Career Advocate. 
        {detection_hint}
        
        INSTRUCTIONS: 
        1. Answer strictly based on the provided context.
        2. If keywords like {detected} are mentioned in the question, find every specific mention of them in the context.
        3. If information is missing, state that clearly but suggest related strengths in AI or Cloud.
        4. Maintain a professional, executive tone.

        CONTEXT: {{context}}
        QUESTION: {{question}}
        """
        
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Retrieving evidence from Zilliz..."):
            try:
                # We use pure semantic retrieval (k=7 for a wider net)
                retriever = v_store.as_retriever(search_kwargs={"k": 7})

                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                
                # Robust Cleaner for Gemini 3.0 content blocks
                if isinstance(full_response, list):
                    full_response = "".join([p.get('text', '') for p in full_response])
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
