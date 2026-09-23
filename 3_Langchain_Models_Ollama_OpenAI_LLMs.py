import os
from dotenv import load_dotenv  # This reads the .env file and makes variables available to Python
load_dotenv()  # This reads the .env file and makes variables available to Python

# Import Ollama - to interact with Ollama cloud LLMs
from langchain_ollama import ChatOllama
# Import OpenAI - to interact with OpenAI cloud LLMs
from langchain_openai import ChatOpenAI

prompt = input("Enter your prompt:\n")

#=============================
# 1 Ollama Example - gpt-oss 
#=============================

llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)
response_ollama = llm_ollama.invoke(prompt)
print("Ollama LLM Response:\n", response_ollama.content)

#================================
# 2 OPENAI Example - gpt-5.4-nano
#================================

llm_openai = ChatOpenAI(model="gpt-5.4-nano", temperature=0.5)
response_openai = llm_openai.invoke(prompt)
print("OpenAI LLM Response:\n", response_openai.content)

#================================
# 3 OPENAI Example - GPT-4.1 Mini
#================================

llm_openai_1 = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
# temp values - 0.0 to 1.0 - to control the randomness of the output. 0.0 is deterministic, 1.0 is random.
response_openai_1 = llm_openai_1.invoke(prompt)
print("OpenAI LLM Response:\n", response_openai_1.content)