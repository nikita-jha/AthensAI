import streamlit as st
import pandas as pd
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
from apify_client import ApifyClient
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from langchain.agents import create_pandas_dataframe_agent
from util.util import search, embedding, chat, scrape, split, summarize
from openai.embeddings_utils import get_embedding, cosine_similarity

name = st.text_input("Opponent Name")
twitter = st.text_input("Opponent Twitter")

# Initialize session state variables
if 'search' not in st.session_state:
    st.session_state['search'] = False

if st.button("search"):
    st.session_state['search'] = True

if st.session_state['search']:
    results = search(name, news=True, time="m", n=5)
    sys = [{
            "role": "system",
            "content": "You are a Democratic opposition research bot. "\
                       "Analyze this passage, identify controversial points or "\
                       "attack-worthy content, and provide specific quotes and "\
                       "a concise summary."
        }]
    summaries = []
    for result in results['news_results']:
        link = result['link']
        text = scrape(link)
        abridged = text[:2000]
        title = result['title']
        summary = chat(abridged, messages = sys, max_tokens=300)
        summaries.append(summary)
        all = ", ".join(summary for summary in summaries)
        if len(all) > 2000:
            all = all[:2000]
    meta = chat("You are a Democratic opposition research bot. Use this information to write a research report of the "
                "opponent's recent activities and key points to attack them on. Use quotes if possible." + all, max_tokens=500, model="gpt-4")
    st.write(meta)
