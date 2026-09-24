import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from the .env file
load_dotenv()

# Define the Ollama local model
model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.5)
output_parser = StrOutputParser()

# ==========================================
# CHAIN 1: Country -> Capital City
# ==========================================
capital_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert. Reply ONLY with the name of the capital city. No punctuation, no extra words."),
    ("user", "What is the capital of {country}?")
])

# Outputs a string (the capital name)
capital_chain = capital_prompt | model | output_parser

# ==========================================
# CHAIN 2: Capital City -> Travel Destinations
# ==========================================
places_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel guide. Provide a bulleted list of exactly 2 most popular places to visit in the given city."),
    ("user", "What are the 2 most popular places to visit in {capital}?")
])

places_chain = places_prompt | model | output_parser

# ==========================================
# INTERCONNECTION: Sequential Execution
# ==========================================
# We use a dictionary mapping with RunnablePassthrough to feed Chain 1's output 
# into Chain 2's expected input key ("capital").
full_sequential_chain = (
    {"capital": capital_chain} 
    | places_chain
)

# ==========================================
# Task-2: Display the Entire Interconnected Graph
# ==========================================
print("\n=== The graphical representation of the interconnected sequential chain ===")
try:
    full_sequential_chain.get_graph().print_ascii()
except Exception as e:
    print(f"Could not print ASCII graph: {e}")

# ==========================================
# Task-3: Execute the Chain
# ==========================================
country_input = "India"
print(f"\nRunning sequential chain for country: **{country_input}**...")
print("\n--- Final Travel Guide Response ---")

# The initial input key matches the first chain's variable {country}
final_output = full_sequential_chain.invoke({"country": country_input})
print(final_output)
