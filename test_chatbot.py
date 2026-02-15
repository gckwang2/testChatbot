import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OracleVS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.title("🤖 Freddy's AI Career Assistant")
st.caption("AI enabled search powered by Oracle keyword+vector, RAG, Google embedding, Gemini Flash LLM")

# --- 2. Connections ---
@st.cache_resource
def get_db_connection():
    # Using thin mode for Oracle DB is usually easier in Streamlit Cloud
    return oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

def init_connections():
    try:
        conn = get_db_connection()
        
        # Embeddings Model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Chat Model - Using standard production naming
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.7 # Recommended for a balance of creativity and accuracy
        )
        
        # Vector Store
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
        template = """
        SYSTEM: Use the following context from Freddy's resume 
        to answer the user's question. If the answer isn't in the context, be honest but 
        highlight related strengths Freddy has.
        
        CONTEXT: {context}
        QUESTION: {question}
        
        INSTRUCTIONS: Summarize Freddy's experience, technical skills, and achievements clearly.
        """
        prompt_template = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        with st.spinner("Searching Freddy's experience..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})

                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                # Execute search and generation
                # RetrievalQA usually maps the input to 'query'
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
                st.info("Check if RESUME_SEARCH has data and your GOOGLE_API_KEY is valid.")
