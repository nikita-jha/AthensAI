import streamlit as st
from census import Census
from us import states

# Input for Census API key
CENSUS_API_KEY = st.text_input("Census API Key", type="password")

# Select state from a list
state_select = st.selectbox("Select a state", [state.name for state in states.STATES])

# Fetch the total population for the selected state
if st.button("Fetch Total Population"):
    if not CENSUS_API_KEY:
        st.warning("Please enter a valid Census API key.")
    else:
        # Instantiate Census object
        c = Census(CENSUS_API_KEY)

        # Get the state FIPS code
        state_fips = states.lookup(state_select).fips

        # Fetch the total population for the selected state
        total_population = c.acs5.state('B01003_001E', state_fips)[0]['B01003_001E']

        # Display the result
        st.write(f"Total population of {state_select}: {total_population}")

import os
import json
import time
import requests
from datetime import date, timedelta
import streamlit as st

# Get the environment variables
AB_CLIENT_UUID = st.text_input("AB_CLIENT_UUID")
AB_CLIENT_SECRET = st.text_input("AB_CLIENT_SECRET", type="password")

if st.button("Generate CSV"):
    BASE_URI = 'https://secure.actblue.com/api/v1'

    # Set up an HTTP session with basic authentication using the ActBlue client uuid and client secret
    session = requests.Session()
    session.auth = (AB_CLIENT_UUID, AB_CLIENT_SECRET)
    session.headers.update({'accept': 'application/json'})

    # Make the Create CSV request to initiate the CSV generation
    body = {
        'csv_type': 'paid_contributions',
        'date_range_start': (date.today() - timedelta(days=31)).isoformat(),
        'date_range_end': (date.today() + timedelta(days=1)).isoformat(),
    }
    response = session.post(f"{BASE_URI}/csvs", json=body)
    response.raise_for_status()
    st.write('Initiated CSV generation successfully')

    # Extract the the CSV ID from the response
    csv = json.loads(response.text)
    csv_id = csv['id']

    # Poll the Get CSV endpoint to check whether the CSV has finished generating
    download_url = None
    while download_url is None:
        time.sleep(1)  # wait briefly between each request to give CSV time to generate
        st.write('Polling for download URL, this may take a few minutes')
        response = session.get(f"{BASE_URI}/csvs/{csv_id}")
        response.raise_for_status()
        csv = json.loads(response.text)
        download_url = csv['download_url']

    # Once we have a download URL, download the file from it
    st.write('Download URL retrieved, downloading CSV')
    filename = 'paid_contributions.csv'
    with open(filename, 'wb') as f:
        response = requests.get(download_url)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    st.write(f"Successfully saved file to {filename}")

from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import openai


def chat(messages, max_tokens=None) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens
    )

    return response.choices[0].message["content"]


def user(content):
  return {
    "role": "user",
    "content": content
  }

def scrape(url):
    response = requests.get(url)

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def split(text, max_length=8192):
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def summarize(text, q):
    if not text:
        return "Error: No text to summarize"

    summaries = []
    chunks = list(split(text))

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        summary = chat([user(chunk + q)])
        summaries.append(summary)

    print(f"Summarized {len(chunks)} chunks.")

    combined = "\n".join(summaries)
    final = chat([user(combined + q)])

    return final


# params = {
#     "q": "Adam Schiff for office",
#     "location": "San Francisco, United States",
#     "hl": "en",
#     "gl": "us",
#     "google_domain": "google.com",
#     "api_key": "key"
# }
#
# search = GoogleSearch(params)
# results = search.get_dict()
# search_dict = []
# links = 0
# q = "\nUsing the above text, if you can, tell me about the candidate's biographical information, platform, " \
#         "main issues, and audience in detail."
# while links < 5:
#     result = results['organic_results'][links]
#     link = result['link']
#     text = scrape(link)
#     if len(text) < 3500:
#         links += 1
#         title = result['title']
#         snippet = result['snippet']
#         summary = summarize(text, q)
#         search_dict.append({'title': title, 'link': link, 'snippet': snippet, 'summary': summary})
#         print(summary)
#     comma = ", "
#     all = comma.join([entry['summary'] for entry in search_dict])
#     meta = summarize(all, q)
#     print(meta)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']