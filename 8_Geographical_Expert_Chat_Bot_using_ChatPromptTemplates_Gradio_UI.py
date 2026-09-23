import os
from dotenv import load_dotenv  # This reads the .env file and makes variables available to Python
load_dotenv()  # This reads the .env file and makes variables available to Python

import gradio as gr
# Import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# Import Ollama - to interact with Ollama cloud LLMs
from langchain_ollama import ChatOllama
# Import OpenAI - to interact with OpenAI cloud LLMs
from langchain_openai import ChatOpenAI

# 1. Define the multi-role template
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise geography expert. Answer in one short sentence."),
    ("human", "What is the capital of {country}?"),
    ("ai", "The capital of {country} is..."),
    ("human", "Please finish the sentence and tell me the name of the capital city.")
])

# Initialize the models
llm_ollama = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.5)
llm_openai = ChatOpenAI(model="gpt-5.4-nano", temperature=0.5)


# 2. Define the core prediction function for Gradio
def get_capital_cities(country_input):
    # Skip empty or whitespace-only inputs
    if not country_input.strip():
        return "Please enter a valid country name.", "Please enter a valid country name."
        
    # Format the template dynamically with the provided country
    formatted_chat_messages = chat_prompt_template.format_messages(country=country_input)

    # Execute Ollama
    try:
        response_ollama = llm_ollama.invoke(formatted_chat_messages)
        ollama_out = response_ollama.content
    except Exception as e:
        ollama_out = f"Ollama Error: {e}"

    # Execute OpenAI
    try:
        response_openai = llm_openai.invoke(formatted_chat_messages)
        openai_out = response_openai.content
    except Exception as e:
        openai_out = f"OpenAI Error: {e}"
        
    return ollama_out, openai_out


# 3. Construct the Gradio App Layout
with gr.Blocks(title="Geography Capital Finder") as demo:
    gr.Markdown("# 🌍 Geography Capital Finder")
    gr.Markdown("Enter a country below to query both **Ollama** and **OpenAI** simultaneously using LangChain Chat Prompt Templates.")
    
    # Input field
    country_textbox = gr.Textbox(label="Enter a country", placeholder="e.g., France, Japan, Brazil...")
    
    # Side-by-side output configuration
    with gr.Row():
        ollama_textbox = gr.Textbox(label="Ollama LLM Response", interactive=False)
        openai_textbox = gr.Textbox(label="OpenAI LLM Response", interactive=False)
        
    # Action Button
    submit_btn = gr.Button("Submit", variant="primary")
    
    # Wire the button click action to the prediction function
    submit_btn.click(
        fn=get_capital_cities, 
        inputs=country_textbox, 
        outputs=[ollama_textbox, openai_textbox]
    )
    
    # Allow pressing "Enter" in the textbox to also trigger submission
    country_textbox.submit(
        fn=get_capital_cities, 
        inputs=country_textbox, 
        outputs=[ollama_textbox, openai_textbox]
    )

# 4. Launch the local web server
if __name__ == "__main__":
    demo.launch(share=True)
