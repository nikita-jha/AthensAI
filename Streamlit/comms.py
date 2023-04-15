import openai
import streamlit as st
from streamlit_chat import message
import os
import re
import time
import pypdf
# import pickledb
import pinecone
import pandas as pd
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain

# Setting page title and header
st.set_page_config(page_title="Campaign Advisor", page_icon=":robot_face:")

PINECONE_API_KEY = os.getenv("PINE_KEY")
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="asia-southeast1-gcp"
)

index_name = "fec"
embeddings = OpenAIEmbeddings()

# Read DB
acc_pinecone = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

# Sidebar - let user choose model and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['model_name'] = []


# generate a response
def generate_response(prompt, model_name):
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    elif model_name == "GPT-4":
        model = "gpt-4"

    qa = ChatVectorDBChain.from_llm(ChatOpenAI(temperature=0, model_name=model), acc_pinecone,
                                    return_source_documents=True)

    result = qa({"question": prompt, "chat_history": ""})
    response = result['answer']

    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input, model_name)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
