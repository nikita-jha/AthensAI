from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

load_dotenv()

app = Flask(__name__)
CORS(app) #This line enables CORS

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    api_key = os.getenv('GPT_API_KEY')  
    generated_text = generate_text(prompt, api_key)
    return jsonify(generated_text)

def generate_text(prompt, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )

    print(response)
    print(response.choices[0].message.content)

if __name__ == '__main__':
    app.run(debug=True)
