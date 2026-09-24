import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load API keys and configurations from your .env file
load_dotenv()

# Initialize the Ollama Cloud model
model = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.7
)

# Task-1 -> Prompt with variable parameter (matching your exact parameter spelling 'contry')
capital_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is the capital of {contry}")
])
country = "India"

# Task-2 -> Define a chain combining the Prompt and Model
simple_chain = capital_prompt | model

# Display the graphical representation of the workflow
print("\n The graphical representation of the chain/runnablesequence: \n")
simple_chain.get_graph().print_ascii()

# Task-3 -> Execute the chain by passing the dictionary payload
print("\n--- Executing Chain ---")
response = simple_chain.invoke({"contry": country})

print("\nModel Response:")
print(response.content)
