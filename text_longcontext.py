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
            system_prompt = f"""
            ROLE: You are Freddy Goh's Senior Career Advocate.
            FULL CAREER DATA:
            {full_context}
            
            USER QUESTION: {prompt}
            """
            
            # --- THE KEY UPDATE IS HERE ---
            raw_response = llm.invoke(system_prompt)
            
            # 1. Check if the response has a 'content' attribute (LangChain style)
            if hasattr(raw_response, 'content'):
                content = raw_response.content
            else:
                content = raw_response

            # 2. Extract text if it's trapped in a list/dictionary (Gemini 3 format)
            if isinstance(content, list) and len(content) > 0:
                # Look for the 'text' key inside the first dictionary
                answer = content[0].get('text', str(content[0]))
            elif isinstance(content, dict):
                answer = content.get('text', str(content))
            else:
                answer = str(content)
            
            # Render the clean answer
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
