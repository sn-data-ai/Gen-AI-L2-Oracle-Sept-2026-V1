import os
from dotenv import load_dotenv  
load_dotenv()  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Define the multi-role template outside the loop so it only compiles once
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Your responses are brief and accurate."),
    MessagesPlaceholder(variable_name="chat_history"), # Fixed spelling here
    HumanMessagePromptTemplate.from_template("{user_input_prompt}")
])

# Initialize the models
llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)

chat_history = []  # Initialize an empty list to store the chat history

print("\nOllama Powered Context Aware Stateful - Chat-bot is Initialized! Type 'exit', 'quit', or 'q' to end the session.\n")

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
    formatted_chat_messages = chat_prompt_template.format_messages(user_input_prompt=user_input_prompt, chat_history=chat_history)

    #=============================
    # Ollama Execution
    #=============================
    try:
        response = llm_ollama.invoke(formatted_chat_messages)
        print("\nOllama LLM Response:")
        print(response.content)
        
        # Context Aware - Stateful 
        # Chat History: Append the current user input and model response to the chat history
        
        chat_history.append(HumanMessage(content=user_input_prompt))
        chat_history.append(AIMessage(content=response.content)) # Changed from SystemMessage to AIMessage
        
    except Exception as e:
        print(f"\nOllama Error: {e}")

        
    print("-" * 40 + "\n")  # Visual separator for the next turn
