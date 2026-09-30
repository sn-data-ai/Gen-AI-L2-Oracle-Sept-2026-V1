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
# STEP 1: Theory, Notes, & Examples Chain
# ==========================================
theory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert SQL instructor. Provide clear theory notes, code examples, and step-by-step explanations for the given topic."),
    ("user", "Explain the SQL topic: {topic}")
])
theory_chain = theory_prompt | model | output_parser

# ==========================================
# STEP 2: Downstream Chains (Rely on generated theory)
# ==========================================

# 2a. MCQ Chain (No Answers)
mcq_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an examiner. Generate exactly 5 Multiple Choice Questions (MCQs) with options (A, B, C, D). "
        "CRITICAL: Base the questions strictly and ONLY on the provided context notes. Do NOT include an answer key, "
        "do NOT reveal the correct answers anywhere, and do NOT include explanations."
    )),
    ("user", "Context Notes:\n{context_notes}\n\nGenerate 5 MCQs based only on the notes above.")
])
mcq_chain = mcq_prompt | model | output_parser

# 2b. Practical Exercises Chain (No Solutions)
practical_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a technical interviewer. Provide exactly 2 practical, real-world database problems/scenarios. "
        "CRITICAL: The scenarios must test concepts covered in the provided context notes. Do NOT include "
        "the solutions, SQL queries, or answer schemas. Provide only the problem statements."
    )),
    ("user", "Context Notes:\n{context_notes}\n\nGenerate 2 practical exercises based on the notes above.")
])
practical_chain = practical_prompt | model | output_parser

# ==========================================
# INTERCONNECTION: Hybrid Sequential & Parallel Pipeline
# ==========================================
# 1. We start by resolving the theory notes.
# 2. We use RunnableParallel to pass those notes into the MCQs and Exercises simultaneously.
hybrid_chain = (
    {"context_notes": theory_chain}  # Step 1: Execute theory chain and save string as 'context_notes'
    | RunnableParallel({
        "theory": lambda x: x["context_notes"],  # Extract the notes string for the final print layout
        "mcqs": mcq_chain,                       # Step 2a: Pipe 'context_notes' into mcq_chain in parallel
        "exercises": practical_chain             # Step 2b: Pipe 'context_notes' into practical_chain in parallel
    })
)

hybrid_chain.get_graph().print_ascii()

# ==========================================
# DYNAMIC WHILE LOOP EXECUTION
# ==========================================
def main():
    print("=== Hybrid Sequential-Parallel SQL Material Generator ===")
    print("Type 'exit' or 'quit' to stop the program.\n")
    
    while True:
        user_input = input("Enter an SQL topic (e.g., Subqueries, Group By, Window Functions): ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting generator. Happy coding!")
            break
            
        print(f"\nProcessing pipeline for '{user_input}'...")
        print(" -> Phase 1: Generating authoritative theory notes...")
        print(" -> Phase 2: Branching parallel evaluation pipelines (Strictly context-bound)...")
        print("Please wait...\n")
        
        try:
            # Invoke the pipeline
            response = hybrid_chain.invoke({"topic": user_input})
            
            # Print the formatted outputs
            print("=" * 60)
            print(f"📖 GENERATED THEORY & NOTES: {user_input.upper()}")
            print("=" * 60)
            print(response["theory"])
            print("\n" + "=" * 60)
            
            print(f"❓ CONTEXT-BOUND MULTIPLE CHOICE QUESTIONS (No Answers)")
            print("=" * 60)
            print(response["mcqs"])
            print("\n" + "=" * 60)
            
            print(f"💻 PRACTICAL EXERCISES (No Solutions)")
            print("=" * 60)
            print(response["exercises"])
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"An error occurred while processing: {e}\n")

if __name__ == "__main__":
    main()
