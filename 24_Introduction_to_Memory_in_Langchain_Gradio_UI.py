import os
from dotenv import load_dotenv
import gradio as gr
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
memory = ConversationBufferMemory(memory_key="history", return_messages=False, verbose=True)

# ==========================================
# 2. STRING-BASED MEMORY PROMPT
# ==========================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, conversational AI assistant. Use the history provided to maintain context."),
    ("user", "Conversation History:\n{history}\n\nCurrent Input: {user_input}")
])

# Helper function to extract history
def get_history_data(x):
    return memory.load_memory_variables({})["history"]

# ==========================================
# 3. INTERCONNECTED EXPRESSION CHAIN
# ==========================================
chat_chain = (
    {
        "history": get_history_data,
        "user_input": RunnablePassthrough()
    }
    | chat_prompt
    | model
    | output_parser
)

# ==========================================
# GRADIO CORE INTERACTION HANDLER
# ==========================================
def respond(message, chat_history):
    if not message.strip():
        return "", chat_history, memory.load_memory_variables({})["history"]
        
    try:
        # 1. Invoke the chain passing just the raw input string
        response = chat_chain.invoke(message)
        
        # 2. Manually save the transaction turn to the legacy memory buffer
        memory.save_context(
            {"input": message}, 
            {"output": response}
        )
        
        # 3. Append to Gradio chat history layout structure
        chat_history.append((message, response))
        
        # 4. Fetch updated memory log text
        verbose_buffer_log = memory.load_memory_variables({})["history"]
        if not verbose_buffer_log:
            verbose_buffer_log = "[Buffer is currently empty]"
            
        return "", chat_history, verbose_buffer_log
        
    except Exception as e:
        chat_history.append((message, f"An error occurred: {e}"))
        return "", chat_history, f"Pipeline Error: {e}"

def clear_session():
    # Clear both the frontend chat and the backend LangChain memory object
    memory.clear()
    return [], "[Memory cleared. Buffer is currently empty]"

# ==========================================
# GRADIO UI STRUCTURE & DESIGN
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="LangChain Verbose Chatbot") as demo:
    gr.Markdown("## 🤖 Context-Aware Chatbot with Live Memory Inspection")
    gr.Markdown("This interface interacts with a dynamic LangChain runtime tracking system powered by legacy `ConversationBufferMemory`.")
    
    with gr.Row():
        # Left Side Column: Interactive Chat Interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat Window", height=450)
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message", 
                    placeholder="Type your prompt here and hit Enter...", 
                    lines=1,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                
            clear_btn = gr.Button("Clear Conversation & Memory", variant="stop")
            
        # Right Side Column: Live Verbose Buffer Output Monitor
        with gr.Column(scale=1):
            verbose_monitor = gr.Textbox(
                label="--- [VERBOSE: CURRENT MEMORY BUFFER CONTENT] ---",
                value="[Buffer is currently empty]",
                lines=21,
                interactive=False,
                show_copy_button=True
            )

    # Wire up the visual event listeners (Triggers on Enter key or Send button click)
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, verbose_monitor])
    submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, verbose_monitor])
    clear_btn.click(clear_session, inputs=None, outputs=[chatbot, verbose_monitor])

# Launch the local application
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
