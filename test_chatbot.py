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

# --- 1. Custom Puter LLM Wrapper for LangChain ---
class PuterLLM(LLM):
    model_name: str = "gpt-5.2" # You can use gpt-4o, gpt-5-nano, etc.

    @property
    def _llm_type(self) -> str:
        return "puter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        async def get_puter_res():
            # Use credentials from st.secrets
            async with PuterClient() as client:
                await client.login(st.secrets["PUTER_USER"], st.secrets["PUTER_PASS"])
                result = await client.ai_chat(
                    prompt=prompt, 
                    options={"model": self.model_name}
                )
                return result["response"]["result"]["message"]["content"]
        return asyncio.run(get_puter_res())

# --- 2. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")
st.title("🤖 Freddy's AI Career Assistant (Puter Edition)")

# --- 3. Connections ---
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
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Initialize Puter LLM instead of Gemini
        llm = PuterLLM(model_name="gpt-5.2")
        
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

# --- 4. Chat UI & Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm now powered by Puter.js (GPT-5.2). Ask about Freddy's skills!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = "CONTEXT: {context}\nQUESTION: {question}\nINSTRUCTIONS: Answer using Freddy's resume."
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
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
            except Exception as e:
                st.error(f"Error: {e}")
