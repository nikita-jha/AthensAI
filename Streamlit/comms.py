import openai
import streamlit as st
import os
import util.util
from util.util import chat
from util.util import search, scrape, split, summarize

# Set up page configuration
st.set_page_config(page_title="comms", page_icon=":mega:", layout="wide")
st.title("comms")
st.write("This tool generates communications for political campaigns using OpenAI's GPT-3 service. "
         "Please enter as much information as you can, and GPT will handle the rest.\n\n"
         "Note: GPT-3 might generate incorrect information, so editing output is still necessary. "
         "This is a demo with limitations.")

# Initialize session state variables
if 'personalize' not in st.session_state:
    st.session_state['personalize'] = False

if 'generate' not in st.session_state:
    st.session_state['generate'] = False

if 'tabs' not in st.session_state:
    st.session_state['tabs'] = ''

if 'types' not in st.session_state:
    st.session_state['types'] = ''

name = st.text_input("Candidate Name: ") + " campaign"
location = st.text_input("Location: ")

# Personalize button
if st.button("Personalize"):
    st.session_state['personalize'] = True
    st.session_state['generate'] = False

# Display additional options if the Personalize button has been clicked
if st.session_state['personalize'] and not st.session_state['generate']:
    results = search(name, location)
    search_dict = []
    links = 0
    q = "\nUsing the text, if you can, tell me about the candidate's biographical information, platform, " \
        "main issues, and audience in detail. Do not just copy verbatim, infer information."

    st.write(q)

    # while links < 2 and links != len(results):
    #     result = results['organic_results'][links]
    #     link = result['link']
    #     text = scrape(link)
    #     if len(text) < 2000:
    #         title = result['title']
    #         snippet = result['snippet']
    #         summary = summarize(text, q)
    #         search_dict.append({'title': title, 'link': link, 'snippet': snippet, 'summary': summary})
    #         all = ", ".join([entry['summary'] for entry in search_dict])
    #     links += 1
    #
    # meta = summarize(all, q)

    # Create a tab selection
    tabs = st.selectbox('Type:', ('Email', 'Social Media', 'Press Release'))
    if tabs == 'Email':
        types = st.selectbox('Type:', ('Fundraising', 'Volunteer', 'Event', 'Other'))
        if types == 'Other':
            types = st.text_input("Details:")
    if tabs == 'Social Media':
        types = st.selectbox('Platform:', ('Twitter', 'Facebook', 'Linkedin', 'Other'))
        if types == 'Other':
            types = st.text_input("Details:")

    # Generate button
    if st.button("Generate"):
        st.session_state['generate'] = True

# Display output if the Generate button has been clicked
if st.session_state['generate']:
    output = chat(
        "hi", model="gpt-4")

    st.write(output)
