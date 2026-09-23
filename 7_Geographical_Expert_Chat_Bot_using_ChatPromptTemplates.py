import os
from dotenv import load_dotenv  # This reads the .env file and makes variables available to Python
load_dotenv()  # This reads the .env file and makes variables available to Python

# Import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# Import Ollama - to interact with Ollama cloud LLMs
from langchain_ollama import ChatOllama
# Import OpenAI - to interact with OpenAI cloud LLMs
from langchain_openai import ChatOpenAI

# 1. Define the multi-role template outside the loop so it only compiles once
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise geography expert. Answer in one short sentence."),
    ("human", "What is the capital of {country}?"),
    ("ai", "The capital of {country} is..."),
    ("human", "Please finish the sentence and tell me the name of the capital city.")
])

# Initialize the models
llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)
llm_openai = ChatOpenAI(model="gpt-5.4-nano", temperature=0.5)

print("\nGeography Chatbot initialized! Type 'exit', 'quit', or 'q' to end the session.\n")

# 2. Continuous Chat Loop
while True:
    # Get the country input from the user
    country_input = input("Enter a country:\n")
    
    # Check for exit commands (case-insensitive and stripped of whitespace)
    if country_input.strip().lower() in ['exit', 'quit', 'q']:
        print("The chat-bot is stopped. Goodbye! Have a great day!")
        break
        
    # Skip empty inputs to prevent API errors
    if not country_input.strip():
        print("Please enter a valid country name.\n")
        continue

    # 3. Format the template dynamically with the current iteration's country
    formatted_chat_messages = chat_prompt_template.format_messages(country=country_input)

    #=============================
    # 1 Ollama Execution
    #=============================
    try:
        response_ollama = llm_ollama.invoke(formatted_chat_messages)
        print("\nOllama LLM Response:")
        print(response_ollama.content)
    except Exception as e:
        print(f"\nOllama Error: {e}")

    #================================
    # 2 OPENAI Execution
    #================================
    try:
        response_openai = llm_openai.invoke(formatted_chat_messages)
        print("\nOpenAI LLM Response:")
        print(response_openai.content)
    except Exception as e:
        print(f"\nOpenAI Error: {e}")
        
    print("-" * 40 + "\n")  # Visual separator for the next turn
