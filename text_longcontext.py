import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="Freddy's Long-Context Advocate", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I have read Freddy's entire consolidated history. Ask me anything!"}]

# --- 2. LOAD THE DATA FROM GITHUB REPO ---
# When deployed on Streamlit Cloud, the 'resume.txt' is sitting in the 
# same root directory as this script. We can read it directly.
@st.cache_data
def load_full_resume():
    # We look for 'resume.txt' which is the standard output of your ingestion script
    filename = "resume.txt"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # If it's not found, we check if you kept the old name
        try:
            with open("freddy_full_history.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "Error: resume.txt not found in GitHub repository. Please run your ingestion script and push the output file to GitHub."

full_context = load_full_resume()

# --- 3. CONNECTION ---
# Using the gemini-3-flash-preview model as requested
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
if prompt := st.chat_input("Query Freddy's consolidated history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching full context..."):
            # The "Long Context Stuffing" method
            system_prompt = f"""
            ROLE: You are Freddy Goh's Senior Career Advocate.
            FULL CAREER DATA:
            {full_context}
            
            USER QUESTION: {prompt}
            
            INSTRUCTION: Use the entire context above to answer. 
            Be specific, reference dates/projects, and be persuasive.
            """
            
            # Simple text extraction logic to avoid metadata errors
            response = llm.invoke(system_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
