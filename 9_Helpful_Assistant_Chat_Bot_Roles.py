import os
from dotenv import load_dotenv  
load_dotenv()  
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

# Langchain
            # 1 Models
            # 2 Prompts
                        # 1 Prompt Templates
                        # 2 ChatPromptTemplates
                        # 3 Messages
                                    # System, Human, AI, MessagePalceHolder

# 1. Define the multi-role template outside the loop so it only compiles once
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Your responses are brief and accurate."),
    HumanMessagePromptTemplate.from_template("{user_input_prompt}")
])


# Initialize the models
llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)

print("\nOllama Chat-bot is Initialized! Type 'exit', 'quit', or 'q' to end the session.\n")

# 2. Continuous Chat Loop
while True:
    # Get the user input
    user_input_prompt = input("\nEnter your prompt here:\n")
    
    # Check for exit commands (case-insensitive and stripped of whitespace)
    if user_input_prompt.strip().lower() in ['exit', 'quit', 'q']:
        print("The chat-bot is stopped. Goodbye! Have a great day!")
        break
        
    # Skip empty inputs to prevent API errors
    if not user_input_prompt.strip():
        print("Please enter a valid prompt.\n")
        continue

    # 3. Format the template dynamically with the current iteration's user input
    formatted_chat_messages = chat_prompt_template.format_messages(user_input_prompt=user_input_prompt)

    #=============================
    # Ollama Execution
    #=============================
    try:
        response = llm_ollama.invoke(formatted_chat_messages)
        print("\nOllama LLM Response:")
        print(response.content)
    except Exception as e:
        print(f"\nOllama Error: {e}")

        
    print("-" * 40 + "\n")  # Visual separator for the next turn
