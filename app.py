import os 
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#enviroment variables for LangSmith Tracking 
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANCHAIN_PROJECT']= "OpenAI ChatBot Project"

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

prompt = ChatPromptTemplate.from_messages([ 
                                           ("system", "You are a helpful AI assistant. Please give a concise answer that is easy to understand."),
                                           ("user", "Question: {question}")
                                           ])

def generate_answer(question, llm, temperature, max_tokens):
    
    llm = ChatGroq(model=llm)
    output_parser = StrOutputParser()
    chain= prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer

st.title("Q&A Chatbot with Groq")
llm = st.sidebar.selectbox("choose your LLM",["gemma2-9b-it", "llama3.1-8b-instant"])

temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.7)

max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=200, value=100)

st.write("Ask a question:")
user_input = st.text_input("question")

if user_input:
    response = generate_answer(user_input, llm, temperature, max_tokens)
    st.write(response)
else:

    st.write("Please enter a question to get an answer.")
