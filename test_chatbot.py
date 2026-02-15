import streamlit as st
import oracledb
import asyncio
from putergenai import PuterClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OracleVS

# UPDATED IMPORTS FOR 2026
# Change these lines in your test_chatbot.py
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA  # <--- Use langchain_classic here
from langchain.llms.base import LLM

# 1. Define the specific prompt for the document answering
system_prompt = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 2. Create the chains
# This chain handles combining the resume snippets into one answer
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

# This chain handles the actual retrieval from Oracle
retriever = v_store.as_retriever(search_kwargs={"k": 5})
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 3. Execute
response = rag_chain.invoke({"input": prompt})
full_response = response["answer"]
