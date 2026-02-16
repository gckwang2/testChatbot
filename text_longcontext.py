import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="Freddy's Long-Context Advocate", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I have read Freddy's entire 200-page history. Ask me anything!"}]

# --- 2. LOAD THE DATA (THE "PULL") ---
# This function reads your 200-page file from your project folder.
@st.cache_data
def load_full_resume():
    try:
        with open("freddy_full_history.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Resume file not found. Please upload freddy_full_history.txt."

full_context = load_full_resume()

# --- 3. CONNECTION ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.2
)

# --- 4. DISPLAY CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. THE LONG-CONTEXT INFERENCE ---
if prompt := st.chat_input("Query Freddy's 200-page history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing full 200-page context..."):
            # Instead of RAG, we "Stuff" the prompt.
            # We put the context FIRST so the model can read it before the question.
            system_prompt = f"""
            ROLE: You are Freddy Goh's Senior Career Advocate.
            FULL CAREER DATA (200 PAGES):
            {full_context}
            
            USER QUESTION: {prompt}
            
            INSTRUCTION: Use the entire context above to answer. 
            Be specific, reference dates/projects, and be persuasive.
            """
            
            response = llm.invoke(system_prompt)
            # Standard cleanup to avoid metadata issues
            answer = response.content if hasattr(response, 'content') else str(response)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
