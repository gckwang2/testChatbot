import streamlit as st
import oracledb
import asyncio
import threading
import nest_asyncio
from putergenai import PuterClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional

# Ensure compatibility with RetrievalQA
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA

# Essential for allowing the Puter loop to run inside a secondary thread
nest_asyncio.apply()

# --- 1. Threaded Puter LLM Wrapper ---
class PuterLLM(LLM):
    model_name: str = "gpt-4o" 

    @property
    def _llm_type(self) -> str:
        return "puter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        def run_async_in_thread(coro):
            """Helper to run async code in a dedicated background thread."""
            result_container = []
            exception_container = []

            def target():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result_container.append(loop.run_until_complete(coro))
                    loop.close()
                except Exception as e:
                    exception_container.append(e)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=30) # 30-second safety timeout

            if exception_container:
                raise exception_container[0]
            if not result_container:
                return "Error: Puter request timed out after 30 seconds."
            return result_container[0]

        async def fetch_response():
            async with PuterClient() as client:
                await client.login(st.secrets["PUTER_USER"], st.secrets["PUTER_PASS"])
                # Fallback to gpt-4o-mini if gpt-4o hangs
                result = await client.ai_chat(
                    prompt=prompt, 
                    options={"model": self.model_name}
                )
                
                # Robust parsing
                if isinstance(result, dict):
                    res_data = result.get("response", {}).get("result", {}).get("message", {})
                    return res_data.get("content", str(result))
                return str(result)

        return run_async_in_thread(fetch_response())

# --- 2. Connections ---
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
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        llm = PuterLLM()
        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH", 
            embedding_function=embeddings
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections()

# --- 3. UI & Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to search Freddy's resume using Puter.js."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. Retrieval Logic ---
if prompt := st.chat_input("Ask about Freddy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = """
        SYSTEM: Use the following context from Freddy's resume to answer the question.
        
        CONTEXT: {context}
        QUESTION: {question}
        
        INSTRUCTIONS: Summarize skills and key achievements.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Puter is thinking..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Search Error: {e}")
