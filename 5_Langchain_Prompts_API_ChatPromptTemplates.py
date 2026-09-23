import os
from dotenv import load_dotenv  # This reads the .env file and makes variables available to Python
load_dotenv()  # This reads the .env file and makes variables available to Python

# Import ChatPromptTemplate (completely removed PromptTemplate)
from langchain_core.prompts import ChatPromptTemplate
# Import Ollama - to interact with Ollama cloud LLMs
from langchain_ollama import ChatOllama
# Import OpenAI - to interact with OpenAI cloud LLMs
from langchain_openai import ChatOpenAI

# 1. Get the country input from the user
country_input = input("Enter a country:\n")

# 2. Define the message-based ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_template("What is the capital of {country}?")

# 3. Format the template -> returns a list of chat messages to use later
formatted_chat_messages = chat_prompt_template.format_messages(country=country_input)

#=============================
# 1 Ollama Example - gpt-oss 
#=============================

llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)
# Pass the structured chat messages to the model
response_ollama = llm_ollama.invoke(formatted_chat_messages)
print("\nOllama LLM Response:\n", response_ollama.content)

#================================
# 2 OPENAI Example - gpt-5.4-nano
#================================

llm_openai = ChatOpenAI(model="gpt-5.4-nano", temperature=0.5)
# Pass the same structured chat messages to the model
response_openai = llm_openai.invoke(formatted_chat_messages)
print("\nOpenAI LLM Response:\n", response_openai.content)
