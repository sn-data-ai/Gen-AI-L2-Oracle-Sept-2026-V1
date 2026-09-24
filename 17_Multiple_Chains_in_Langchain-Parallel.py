import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

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
# INTERCONNECTION: Sequential with Memory Preservation
# ==========================================
# 1. First, we take the initial input and pass it to capital_chain.
# 2. We use RunnableParallel (via a dict) to keep the capital string AND pass it to Chain 2.
# 3. Finally, we output a dictionary containing both the capital name and the tourist places.
full_sequential_chain = (
    {"capital": capital_chain}  # Step 1: Generate the capital name from the country input
    | RunnableParallel({
        "capital_city": RunnablePassthrough(),  # Passes the capital string straight through to the final output
        "tourist_places": places_chain          # Pipes the capital string directly into the places chain
    })
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
print("\n--- Final Structured Response ---")

# The final output is now a clean dictionary containing both elements
final_output = full_sequential_chain.invoke({"country": country_input})

print(f"Capital City: {final_output['capital_city']}")
print(f"Top Places to Visit:\n{final_output['tourist_places']}")
