import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus 
from langchain_core.messages import AIMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's AI Career Assistant. Ask me anything, and I will now show you the similarity scores for my sources!"}
    ]

st.title("🤖 Freddy's AI Career Assistant")
st.caption("Engine: Zilliz Cloud | Score Tracking Enabled")

# --- 2. Connection Logic ---
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

# --- 3. Sidebar ---
with st.sidebar:
    st.header("Engine Settings")
    available_models = ["Gemini 3 Flash (Direct Google)", "Gemini 2.5 Pro (Direct Google)", "Qwen 3 Max Thinking (Alibaba)", "GPT-OSS-120B (Direct Groq)", "Llama 3.3 70B (Direct Groq)"]
    model_choice = st.selectbox("Select AI Engine:", options=available_models)
    v_store, llm_or_err = init_connections(model_choice)
    
    if st.button("Clear Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat reset."}]
        st.rerun()

# --- 4. Render History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. Main Interaction Logic (With Score Tracking) ---
if prompt := st.chat_input("Ask about Freddy's experience"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if v_store is None:
            st.error("Database offline.")
        else:
            with st.spinner("Searching and calculating similarity scores..."):
                try:
                    # 🟢 THE CHANGE: Use similarity_search_with_score
                    # This returns a list of tuples: (Document, Score)
                    docs_with_scores = v_store.similarity_search_with_score(prompt, k=5)

                    context_entries = []
                    score_debug = []

                    for doc, score in docs_with_scores:
                        source_name = doc.metadata.get('file_name', 'Resume')
                        # Round score to 4 decimal places for readability
                        rounded_score = round(float(score), 4)
                        
                        context_entries.append(f"SOURCE: {source_name} (Similarity: {rounded_score})\nCONTENT: {doc.page_content}")
                        score_debug.append(f"- {source_name}: {rounded_score}")

                    context_text = "\n\n---\n\n".join(context_entries)
                    
                    system_msg = f"SYSTEM: Answer using this context.\n\nCONTEXT:\n{context_text}\n\nQUESTION: {prompt}"

                    # Invoke Model
                    raw_res = llm_or_err.invoke(system_msg)
                    
                    # Clean Response
                    if hasattr(raw_res, 'content'):
                        txt = raw_res.content
                        clean_text = "".join([p['text'] for p in txt if 'text' in p]) if isinstance(txt, list) else str(txt)
                    else:
                        clean_text = str(raw_res)

                    # 🟢 Append Score Data to the bottom of the answer
                    score_footer = "\n\n**Search Accuracy (Top 5 Matches):**\n" + "\n".join(score_debug)
                    final_answer = clean_text + score_footer

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                except Exception as e:
                    st.error(f"Error: {e}")
