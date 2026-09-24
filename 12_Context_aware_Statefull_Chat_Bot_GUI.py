import os
from dotenv import load_dotenv  
load_dotenv()  
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Initialize LangChain Components
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Your responses are brief and accurate."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{user_input_prompt}")
])

llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)

# Stateful storage for current session history
session_chat_history = []

def chat_engine(user_input_prompt):
    """Processes the input through Ollama and updates the chat history matrix."""
    global session_chat_history
    
    if not user_input_prompt.strip():
        return "Please enter a valid prompt.", get_formatted_history_string()
        
    try:
        # Format and invoke the LLM
        formatted_chat_messages = chat_prompt_template.format_messages(
            user_input_prompt=user_input_prompt, 
            chat_history=session_chat_history
        )
        response = llm_ollama.invoke(formatted_chat_messages)
        ai_response = response.content
        
        # Append instances to session history state
        session_chat_history.append(HumanMessage(content=user_input_prompt))
        session_chat_history.append(AIMessage(content=ai_response))
        
        # Return the latest response and the full debug string
        return ai_response, get_formatted_history_string()
        
    except Exception as e:
        return f"Ollama Error: {e}", get_formatted_history_string()

def get_formatted_history_string():
    """Generates a scannable text readout showing object states."""
    if not session_chat_history:
        return "Chat history is empty."
        
    output_lines = []
    # Step by 2 to capture the pair structure
    for index, i in enumerate(range(0, len(session_chat_history), 2), start=1):
        user_msg = session_chat_history[i]
        ai_msg = session_chat_history[i+1] if (i+1) < len(session_chat_history) else None
        
        output_lines.append(f"🔄 Interaction Turn #{index}:")
        output_lines.append(f" 👤 Appended: {type(user_msg).__name__} -> content='{user_msg.content}'")
        if ai_msg:
            output_lines.append(f" 🤖 Appended: {type(ai_msg).__name__} -> content='{ai_msg.content}'\n")
            
    return "\n".join(output_lines)

def clear_session():
    """Flushes state history."""
    global session_chat_history
    session_chat_history = []
    return "", "", "Chat history is empty."

# 2. Build tailored Gradio Interface
with gr.Blocks(title="Ollama Context-Aware Chatbot") as demo:
    gr.Markdown("# 🤖 Ollama Context-Aware Stateful Assistant")
    gr.Markdown("Type your prompt below. The application passes state to the LLM and tracks exact internal object changes.")
    
    with gr.Row():
        # Left Side: Chat Inputs & Outputs
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="Your Prompt Input", 
                placeholder="Ask me something...", 
                lines=3, # Tall input box for typing flexibility
                max_lines=6
            )
            submit_btn = gr.Button("Send to Ollama", variant="primary")
            
            output_box = gr.Textbox(
                label="Latest Ollama LLM Response", 
                lines=8, # Deep container optimized for receiving structured responses
                interactive=False
            )
            clear_btn = gr.Button("Clear Conversation State", variant="stop")
            
        # Right Side: Real-time Append Monitor Box
        with gr.Column(scale=2):
            history_debug_box = gr.Textbox(
                label="Append Monitor (What is currently inside LangChain chat_history)",
                value="Chat history is empty.",
                lines=15, # Maximized screen real estate to easily scroll past long message vectors
                interactive=False
            )

    # Wire event logic strings
    submit_btn.click(
        fn=chat_engine, 
        inputs=[user_input], 
        outputs=[output_box, history_debug_box]
    )
    
    # Optional shortcut: trigger on Enter key inside textbox
    user_input.submit(
        fn=chat_engine, 
        inputs=[user_input], 
        outputs=[output_box, history_debug_box]
    )
    
    clear_btn.click(
        fn=clear_session,
        inputs=[],
        outputs=[user_input, output_box, history_debug_box]
    )

# 3. Launch App Local Web Server
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
