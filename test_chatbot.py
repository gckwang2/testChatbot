import streamlit as st
import oracledb
import asyncio
import nest_asyncio
from putergenai import PuterClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS

# --- 2026 MODULAR IMPORTS ---
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional

# Using langchain_classic to maintain compatibility with your RetrievalQA structure
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA

# Apply nest_asyncio to allow Puter's async calls to run within Streamlit
nest_asyncio.apply()

# --- 1. Custom Puter LLM Wrapper ---
class PuterLLM(LLM):
    model_name: str = "gpt-4o" 

    @property
    def _llm_type(self) -> str:
        return "puter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        async def fetch_response():
            try:
                async with PuterClient() as client:
                    # Logs into Puter using secrets
                    await client.login(st.secrets["PUTER_USER"], st.secrets["PUTER_PASS"])
                    
                    # Request generation from Puter
                    result = await client.ai_chat(
                        prompt=prompt, 
                        options={"model": self.model_name}
                    )
                    
                    # 2026 Robust Parsing
                    if isinstance(result, dict):
                        # Attempt to find content in common nested paths
                        res_obj = result.get("response", {}).get("result", {}).get("message", {})
                        content = res_obj.get("content")
                        if content:
                            return content
                        return str(result)
                    return str(result)
            except Exception as e:
                return f"Puter Error: {str(e)}"

        # Run the async function in the current loop
        return asyncio.run(fetch_response())

# --- 2. Database & App Initialization ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

@st.cache_resource
def get_db_connection():
    return oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

def init_connections():
    try:
        conn = get_db_connection()
        conn.ping()
        
        # Google Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Puter LLM instance
        llm = PuterLLM()
        
        # Oracle Vector Store
        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH", 
            embedding_function=embeddings
        )
        return v_store, llm, conn
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm, conn = init_connections()

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to search Freddy's resume. How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Retrieval Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Your specific Prompt Template logic
        template = """
        SYSTEM: Use the following context from Freddy's resume 
        to answer the user's question. If the answer isn't in the context, be honest but 
        highlight related strengths Freddy has.
        
        CONTEXT: {context}
        QUESTION: {question}
        
        INSTRUCTIONS: Summarize Freddy's experience, specific technical skills, and key achievements.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Searching Freddy's experience..."):
            try:
                # Set up the retriever
                retriever = v_store.as_retriever(search_kwargs={"k": 5})

                # Create the QA Chain
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                # Execute search
                # RetrievalQA with custom prompt expects 'query' key
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
                st.info("Check if the table RESUME_SEARCH contains data and valid vectors.")
