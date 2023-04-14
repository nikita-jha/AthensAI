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

meta=""

personalize = st.checkbox('Personalize?')
if personalize:
    name = st.text_input("Candidate Name: ") + " campaign"
    location = st.text_input("Location: ")
    if st.button("Personalize"):
        results = search(name, location)
        search_dict = []
        links = 0
        q = "\nUsing the text, if you can, tell me about the candidate's biographical information, platform, " \
            "main issues, and audience in detail. Do not just copy verbatim, infer information."

        while links < 3 and links != len(results):
            result = results['organic_results'][links]
            link = result['link']
            text = scrape(link)
            if len(text) < 3000:
                title = result['title']
                snippet = result['snippet']
                summary = summarize(text, q)
                search_dict.append({'title': title, 'link': link, 'snippet': snippet, 'summary': summary})
                st.write(summary)
                all = ", ".join([entry['summary'] for entry in search_dict])
            links += 1

        meta = summarize(all, q)
        st.write(meta)

# Create a tab selection
tabs = st.selectbox(
    'Which communication do you want to create? 📄',
    ('Email 📧', 'Else'))


# Function to generate a tweet
def tweet(output):
    return generic_completion(
        "Generate a tweet summarizing the following text. "
        "Make it engaging and concise: " + output)


# Email tab
if tabs == 'Email 📧':
    subject = st.text_input("Email subject:")

    if st.button(label="Generate Email"):
        try:
            output = chat("Write an engaging email for a political campaign. Make sure it is not repetitive and from "
                          "candidate's perspective. Use these " \
                          "details if helpful:" + meta, model="gpt-4")
            st.write("```")
            st.write(output)
            st.write("```")
        except:
            st.write("An error occurred while processing your request.")
