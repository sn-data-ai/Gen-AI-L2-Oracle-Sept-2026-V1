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
# INTERCONNECTION: Sequential with Memory Preservation (FIXED)
# ==========================================
full_sequential_chain = (
    # 1. First, generate a dictionary containing {"capital": "<city_name>"}
    {"capital": capital_chain}  
    
    # 2. Use RunnableParallel to correctly map the inputs and outputs
    | RunnableParallel({
        # Extracts just the string value of "capital" for the final output dictionary
        "capital_city": lambda x: x["capital"],  
        
        # Passes the {"capital": ...} dictionary smoothly into places_chain
        "tourist_places": places_chain          
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
# Task-3: Dynamic Execution Loop
# ==========================================
print("\n" + "="*50)
print("Geography & Travel Guide Chain Initialized.")
print("Type 'exit' or 'quit' to stop the program.")
print("="*50 + "\n")

while True:
    # 1. Take dynamic user input
    user_input = input("Enter a country name: ").strip()
    
    # 2. Check for exit/quit condition
    if user_input.lower() in ['exit', 'quit']:
        print("\nExiting program. Goodbye!")
        break
        
    # 3. Handle empty inputs gracefully
    if not user_input:
        print("Input cannot be empty. Please enter a country name.\n")
        continue

    print(f"\nRunning sequential chain for country: **{user_input}**...")
    print("\n--- Final Structured Response ---")
    
    try:
        # 4. Invoke the chain with the dynamic input
        final_output = full_sequential_chain.invoke({"country": user_input})
        
        # 5. Print out the structured results
        print(f"Capital City: {final_output['capital_city']}")
        print(f"Top Places to Visit:\n{final_output['tourist_places']}")
        print("-" * 40 + "\n")
        
    except Exception as e:
        print(f"An error occurred while processing the request: {e}\n")
