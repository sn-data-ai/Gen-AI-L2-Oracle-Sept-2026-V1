import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Initialize the Ollama Cloud model
model = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.7
)

# Task-1 -> Prompt
# Note: from_messages expects a list of tuples/messages. 
# We use a simple system/human structure or a basic template.
capital_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is the capital of India")
])

# Task-2 -> Define a chain using the pipe operator
simple_chain = capital_prompt | model

# Print the types
print("\n--- Component Types ---")
print(f"Prompt Type: {type(capital_prompt)}")
print(f"Chain Type:  {type(simple_chain)}")

# Display the graphical representation
print("\n The graphical representation of the chain/runnablesequence: \n")
simple_chain.get_graph().print_ascii()
