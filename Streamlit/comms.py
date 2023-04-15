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

if 'generated' not in st.session_state:
    st.session_state['generated'] = False

name = st.text_input("Candidate Name: ") + " campaign issues bio"
# Personalize button
if st.button("Personalize"):
    st.session_state['personalize'] = True
    st.session_state['generate'] = False

# Display additional options if the Personalize button has been clicked
if st.session_state['personalize']:
    if not st.session_state['generate']:
        st.write("Personalizing! This should take under a minute.")
    results = search(name)
    summaries = []
    q = "\nTell me about the candidate's biography, platform, " \
        "and main issues. Keep it concise but specific."

    meta = ""
    for result in results['organic_results']:
        link = result['link']
        text = scrape(link)
        if len(text) < 4000:
            title = result['title']
            snippet = result['snippet']
            summary = chat(text + q, max_tokens=200)
            summaries.append(summary)
            st.write(summary)

    all = ", ".join(summary for summary in summaries)
    meta = chat(all + q, max_tokens=200)

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
if st.session_state['generate'] and not st.session_state['generated']:
    st.write(meta)
    output = chat(
        f"Generate an engaging, long, and unrepetitive {tabs} for {types} in the perspective of candidate. Use this info, if relevant {meta}.")
    st.session_state['generated'] = True

if st.session_state['generated']:
    st.write(output)
