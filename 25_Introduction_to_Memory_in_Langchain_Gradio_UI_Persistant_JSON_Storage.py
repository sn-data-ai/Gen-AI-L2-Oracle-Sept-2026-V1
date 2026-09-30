import os
import json
from dotenv import load_dotenv
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict, messages_to_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Load environment variables from the .env file
load_dotenv()

# Define the Ollama local model
model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)
output_parser = StrOutputParser()

# Path to the persistent storage file
HISTORY_FILE = "chat_history.json"

# ==========================================
# 1. FIXED FILE STORAGE HANDLERS
# ==========================================
def load_stored_history():
    """Loads message history from a local JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Reconstructs proper HumanMessage and AIMessage objects from JSON strings
            return messages_from_dict(data)
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def save_to_stored_history(messages):
    """Saves raw message history directly to a local JSON file."""
    try:
        # Converts LangChain message structures into standard JSON serializable dicts
        serializable_data = messages_to_dict(messages)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
    except Exception as e:
        print(f"Error saving file: {e}")

# ==========================================
# 2. MODERN CONTEXT-AWARE PROMPT ARCHITECTURE
# ==========================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, conversational AI assistant. Refer to the chat history provided to maintain strict context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{user_input}")
])

chat_chain = chat_prompt | model | output_parser

# ==========================================
# GRADIO CORE INTERACTION HANDLER
# ==========================================
def respond(message, chat_history):
    if not message.strip():
        # Fetch unchanged current log state
        current_disk_log = json.dumps(messages_to_dict(load_stored_history()), indent=2)
        return "", chat_history, current_disk_log
        
    try:
        # 1. Pull the absolute latest historical state straight from persistent disk storage
        saved_messages = load_stored_history()
        
        # 2. Invoke the chain passing the live file structure
        response = chat_chain.invoke({
            "chat_history": saved_messages,
            "user_input": message
        })
        
        # 3. Append the new conversation turns to our history tracker list
        saved_messages.append(HumanMessage(content=message))
        saved_messages.append(AIMessage(content=response))
        
        # 4. Save the updated list securely back to the file system
        save_to_stored_history(saved_messages)
        
        # 5. Append to Gradio frontend chat layout object
        chat_history.append((message, response))
        
        # 6. Format the raw JSON file content to show live in the debug panel
        raw_disk_data = json.dumps(messages_to_dict(saved_messages), indent=2)
            
        return "", chat_history, raw_disk_data
        
    except Exception as e:
        chat_history.append((message, f"An error occurred: {e}"))
        return "", chat_history, f"Pipeline Storage Error: {e}"

def clear_session():
    """Wipes out both the UI layout and deletes the disk storage file safely."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return [], "[Persistent JSON file deleted. Storage is now empty]"

# ==========================================
# INITIAL LOAD FOR PERSISTENT UI STATE
# ==========================================
# When restarting the app, populate the UI immediately from whatever was previously saved to disk
initial_messages = load_stored_history()
initial_gradio_history = []
for i in range(0, len(initial_messages), 2):
    if i + 1 < len(initial_messages):
        initial_gradio_history.append((initial_messages[i].content, initial_messages[i+1].content))

initial_raw_json = json.dumps(messages_to_dict(initial_messages), indent=2) if initial_messages else "[Storage is empty]"

# ==========================================
# GRADIO UI STRUCTURE & DESIGN
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="Stable Storage Chatbot") as demo:
    gr.Markdown("## 💾 Chatbot with Persistent Local File Storage")
    gr.Markdown("This interface reads and writes conversation history directly from `chat_history.json`. Closing the app or server will not erase the memory.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(value=initial_gradio_history, label="Chat Window", height=450)
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message", 
                    placeholder="Type your prompt here and hit Enter...", 
                    lines=1,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                
            clear_btn = gr.Button("Delete Persistent File & Clear History", variant="stop")
            
        with gr.Column(scale=1):
            verbose_monitor = gr.Textbox(
                label="--- [LIVE DISK STORAGE CONTROLLER: chat_history.json] ---",
                value=initial_raw_json,
                lines=21,
                interactive=False,
                show_copy_button=True
            )

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, verbose_monitor])
    submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, verbose_monitor])
    clear_btn.click(clear_session, inputs=None, outputs=[chatbot, verbose_monitor])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
    #demo.launch(share=True)
