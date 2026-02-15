import streamlit as st
import oracledb
import asyncio
from putergenai import PuterClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS

# --- UPDATED IMPORTS FOR 2026 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Any, List, Optional

# ... (Previous Connection Logic for PuterLLM and get_db_connection) ...

# --- 4. Chat Input & Retrieval Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Define the specific prompt for document answering
        system_prompt = (
            "Use the following pieces of retrieved context from Freddy's resume "
            "to answer the user's question. If the answer isn't in the context, "
            "be honest but highlight related strengths Freddy has."
            "\n\n"
            "CONTEXT: {context}"
        )

        # 2026 Modern Chat Template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        with st.spinner("Searching Freddy's experience..."):
            try:
                # 2. Setup the Retriever
                retriever = v_store.as_retriever(search_kwargs={"k": 5})

                # 3. Create the Modern RAG Chain
                # Handles combining documents into the prompt
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                
                # Handles the actual retrieval and passing to the LLM
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # 4. Execute
                # In 2026, 'invoke' is the standard; result is a dict with 'answer'
                response = rag_chain.invoke({"input": prompt})
                full_response = response["answer"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
