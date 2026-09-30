import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# Load environment variables from the .env file
load_dotenv()

# Define the Ollama local models
model_router = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.0) # Low temp for strict routing
model_expert = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)

output_parser = StrOutputParser()

# ==========================================
# ROUTER CHAIN: Evaluates the classification
# ==========================================
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support router. Classify the query into exactly one of these categories: 'COMPLAINT' or 'SALES'. Reply with ONLY the category name."),
    ("user", "Classify this customer query: {query}")
])

# Outputs a string classification ('COMPLAINT' or 'SALES')
router_chain = router_prompt | model_router | output_parser

# ==========================================
# EXPERT CHAIN 1: Handling Complaints
# ==========================================
complaint_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer retention specialist. Write a polite, empathetic apology letter offering a 20% discount code (SORRY20). Keep it brief."),
    ("user", "Address this complaint: {query}")
])

complaint_chain = complaint_prompt | model_expert | output_parser

# ==========================================
# EXPERT CHAIN 2: Handling Sales / Inquiries
# ==========================================
sales_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enthusiastic sales representative. Highlight the top 2 benefits of our product and invite them to schedule a free demo call. Keep it concise."),
    ("user", "Respond to this product inquiry: {query}")
])

sales_chain = sales_prompt | model_expert | output_parser

# ==========================================
# INTERCONNECTION: Conditional Branching
# ==========================================
# RunnableBranch takes a list of (condition_function, runnable) tuples, 
# and a final default runnable if no conditions are met.
conditional_branch = RunnableBranch(
    (lambda x: "COMPLAINT" in x["topic"].upper(), complaint_chain),
    (lambda x: "SALES" in x["topic"].upper(), sales_chain),
    sales_chain # Default fallback
)

# Combine the router step and the branching step together
full_conditional_chain = (
    {"topic": router_chain, "query": lambda x: x["query"]}
    | conditional_branch
)

# ==========================================
# Task-2: Display the Entire Interconnected Graph
# ==========================================
print("\n=== The graphical representation of the interconnected conditional chain ===")
try:
    # Note: Conditional branches show routing blocks in the visual graph
    full_conditional_chain.get_graph().print_ascii()
except Exception as e:
    print(f"Could not print ASCII graph: {e}")

# ==========================================
# Task-3: Execute the Chain
# ==========================================
user_query = input("Enter a customer support message/query: ")
print(f"\nRunning conditional chain for query: **{user_query}**...")

# We can peek at how the router classified it first
detected_topic = router_chain.invoke({"query": user_query}).strip()
print(f'\n The router classified this query as: {detected_topic}')
print("\n--- Final Specialized Team Response ---")

# Execute the complete branch logic. The input key matches {query}
final_output = full_conditional_chain.invoke({"query": user_query})
print(final_output)
