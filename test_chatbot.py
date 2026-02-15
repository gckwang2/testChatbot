import streamlit as st
import oracledb
import asyncio
from putergenai import PuterClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS

# --- UPDATED IMPORTS FOR 2026 MODULARITY ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional

# In 2026, many chains have moved to langchain_classic or specific sub-paths
try:
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. Custom Puter LLM Wrapper ---
class PuterLLM(LLM):
    model_name: str = "gpt-4o" 

    @property
    def _llm_type(self) -> str:
        return "puter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        async def fetch_response():
            async with PuterClient() as client:
                await client.login(st.secrets["PUTER_USER"], st.secrets["PUTER_PASS"])
                result = await client.ai_chat(
                    prompt=prompt, 
                    options={"model": self.model_name}
                )
                return result["response"]["result"]["message"]["content"]
        return asyncio.run(fetch_response())

# --- 2. Page Config & Connections ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

@st.cache_resource
def get_db_connection():
    return oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

def init_app():
    try:
        conn = get_db_connection()
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

v_store, llm = init_app()

# --- 3. Chat Session State & UI ---
st.title("🤖 Freddy's AI Career Assistant")
st.caption("Powered by Puter.js (GPT-4o) & Oracle Vector Search")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can now search Freddy's resume using AI semantic matching. Ask me anything!"}
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
        # System instructions
        system_prompt = (
            "Use the following pieces of retrieved context from Freddy's resume "
            "to answer the question. If the answer isn't in the context, be honest but "
            "highlight related strengths Freddy has."
            "\n\n"
            "CONTEXT: {context}"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        with st.spinner("Searching Freddy's experience..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})

                # Modern RAG Chain creation
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Execute
                response = rag_chain.invoke({"input": prompt})
                full_response = response["answer"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
