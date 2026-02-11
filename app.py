import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Set up embeddings (must match ingest.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vectorstore
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Groq LLM setup
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Define Prompt Template (Updated for ChatPromptTemplate)
template = """You are an assistant for Mysoft Heaven (BD) Ltd. Answer only based on the company information provided. If you don't know the answer, say you don't know.
Context: {context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Modern LCEL Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit App
st.title("Mysoft Heaven AI Assistant")
user_input = st.text_input("Ask about Mysoft Heaven:")

if user_input:
    with st.spinner("Thinking..."):
        # Invoke the chain
        response = chain.invoke(user_input)
        st.write(response)
