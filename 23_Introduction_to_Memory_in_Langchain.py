# Models
# Prompts
        # Prompt Templates
        # Chat Prompt Templates
        # Messages - MessgaePlaceHolders, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

        # Example - Context Aware Chat bot - Python list variables 
# Chains
        # Simple
        # Sequential
        # Parallel
        # Custom
        # Conditional
# Memory
        # Context Aware Chat-Bot 

        # Legacy Memory - 
        # ConversationBufferMemory, -> Entire conversation
        # ConversationBufferWindowMemory, -> Last N messages / last few interactions
        # ConversationEntityMemory -> Conversation Summary


import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from the .env file
load_dotenv()

# Define the Ollama local model
model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)
output_parser = StrOutputParser()

# ==========================================
# 1. INITIALIZE LEGACY MEMORY WITH VERBOSE LOGGING
# ==========================================
# Setting verbose=True logs internal state actions
memory = ConversationBufferMemory(memory_key="history", return_messages=False, verbose=False)

# ==========================================
# 2. STRING-BASED MEMORY PROMPT
# ==========================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, conversational AI assistant. Use the history provided to maintain context."),
    ("user", "Conversation History:\n{history}\n\nCurrent Input: {user_input}")
])

# Helper function to extract history and print it out to simulate full verbose tracking
def get_and_print_history(x):
    history_data = memory.load_memory_variables({})["history"]
    print("\n--- [VERBOSE: CURRENT MEMORY BUFFER CONTENT] ---")
    print(history_data if history_data else "[Buffer is currently empty]")
    print("------------------------------------------------\n")
    return history_data

# ==========================================
# 3. INTERCONNECTED EXPRESSION CHAIN
# ==========================================
chat_chain = (
    {
        "history": get_and_print_history,
        "user_input": RunnablePassthrough()
    }
    | chat_prompt
    | model
    | output_parser
)

# ==========================================
# DYNAMIC WHILE LOOP EXECUTION
# ==========================================
def main():
    print("=== Verbose Legacy ConversationBufferMemory Chatbot ===")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        user_message = input("You: ").strip()
        
        if not user_message:
            continue
            
        if user_message.lower() in ['exit', 'quit']:
            print("AI: Goodbye! Have a great day!")
            break
            
        try:
            # 1. Invoke the chain passing just the raw input string
            response = chat_chain.invoke(user_message)
            
            print(f"AI: {response}\n")
            
            # 2. Manually save the transaction turn to the memory buffer
            memory.save_context(
                {"input": user_message}, 
                {"output": response}
            )
            
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    main()
