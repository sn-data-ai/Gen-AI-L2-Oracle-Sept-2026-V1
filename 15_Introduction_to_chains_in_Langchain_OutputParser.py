import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Use the Ollama LLM integration from langchain_community
from langchain_community.chat_models import ChatOllama

# Load environment variables from the .env file
load_dotenv()

# Task-1 -> Prompt setup (Fixed the typo 'contry' to 'country')
capital_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "What is the capital of {country}? Generate only the name of the city.")
])
country_input = "India"

# Task-2 -> Model Definition using Ollama
# Replace 'llama3' with the exact name of the OSS model you have downloaded locally
model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.5)

# Task-2 -> Define the chain
capital_name = StrOutputParser()
simple_chain = capital_prompt | model | capital_name

# Print the graphical representation of the chain
print("\nThe graphical representation of the chain/runnablesequence:\n")
try:
    simple_chain.get_graph().print_ascii()
except Exception as e:
    print(f"Could not print ASCII graph: {e}")

# Task-3 -> Invoke the chain
print("\n--- Model Response ---")
response = simple_chain.invoke({"country": country_input})
print(response)
