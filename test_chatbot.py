import streamlit as st
import oracledb
import asyncio
from putergenai import PuterClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, List, Optional

# --- 1. Custom Puter LLM Wrapper ---
class PuterLLM(LLM):
    # Using the 2026 standard for Puter's high-end reasoning model
    model_name: str = "gpt-4o" 

    @property
    def _llm_type(self) -> str:
        return "puter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        async def fetch_response():
            async with PuterClient() as client:
                # Logs into your Puter account to use your free credits
                await client.login(st.secrets["PUTER_USER"], st.secrets["PUTER_PASS"])
                result = await client.ai_chat(
                    prompt=prompt, 
                    options={"model": self.model_name}
                )
                return result["response"]["result"]["message"]["content"]
        return asyncio.run(fetch_response())

# --- 2. Connections & Initialization ---
@st.cache_resource
def init_db():
    return oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

def init_app():
    conn = init_db()
    # We still use Google for embeddings as they are very efficient for RAG
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

v_store, llm = init_app()

# --- 3. Streamlit UI ---
st.title("🤖 Freddy's Career Assistant (Puter + Oracle)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm now powered by Puter.js. Ask me about Freddy's experience!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("What are Freddy's top technical skills?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Puter is matching resume data..."):
            retriever = v_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            response = qa_chain.invoke({"query": prompt})
            st.write(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
