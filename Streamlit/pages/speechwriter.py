import openai
import streamlit as st

openai.api_key = st.secrets["oai-key"]
os.environ["OPENAI_API_KEY"] = st.secrets["oai-key"]

def generic_completion(prompt):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85
    )
    message = completions['choices'][0]['message']['content']
    return message.strip()