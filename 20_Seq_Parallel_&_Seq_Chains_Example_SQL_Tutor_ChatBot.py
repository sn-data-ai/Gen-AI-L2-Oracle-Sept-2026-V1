import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableParallel

# Load environment variables from the .env file
load_dotenv()

# Define the Ollama local model
model = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.5)
output_parser = StrOutputParser()

# ==========================================
# PHASE 1: Theory, Notes, & Examples Chain
# ==========================================
theory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert SQL instructor. Provide clear theory notes, code examples, and step-by-step explanations for the given topic."),
    ("user", "Explain the SQL topic: {topic}")
])
theory_chain = theory_prompt | model | output_parser

# ==========================================
# PHASE 2: Questions Generation Chains
# ==========================================

# 2a. MCQ Generation (Strictly No Answers)
mcq_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an examiner. Generate exactly 5 Multiple Choice Questions (MCQs) with options (A, B, C, D). "
        "CRITICAL: Base the questions strictly and ONLY on the provided context notes. Do NOT include an answer key, "
        "do NOT reveal the correct answers anywhere, and do NOT include explanations."
    )),
    ("user", "Context Notes:\n{context_notes}\n\nGenerate 5 MCQs based only on the notes above.")
])
mcq_gen_chain = mcq_gen_prompt | model | output_parser

# 2b. Exercise Generation (Strictly No Solutions)
exercise_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a technical interviewer. Provide exactly 2 practical, real-world database problems/scenarios. "
        "CRITICAL: The scenarios must test concepts covered in the provided context notes. Do NOT include "
        "the solutions, SQL queries, or answer schemas. Provide only the problem statements."
    )),
    ("user", "Context Notes:\n{context_notes}\n\nGenerate 2 practical exercises based on the notes above.")
])
exercise_gen_chain = exercise_gen_prompt | model | output_parser

# ==========================================
# PHASE 3: Solver / Answer Key Chains
# ==========================================

# 3a. MCQ Solver Chain
mcq_solve_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert student tracking system. Provide a clean, direct answer key for the provided MCQs. State the question number and the correct option letter, followed by a one-sentence verification."),
    ("user", "Here are the MCQs:\n\n{mcqs}\n\nProvide the correct answers for these questions.")
])
mcq_solve_chain = mcq_solve_prompt | model | output_parser

# 3b. Exercise SQL Solution Generator Chain
exercise_solve_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert database administrator. Provide production-ready, highly optimized SQL query solutions for the given practical exercises. Include a brief description of why the approach works."),
    ("user", "Here are the practical exercises:\n\n{exercises}\n\nProvide the functional SQL query solutions for them.")
])
exercise_solve_chain = exercise_solve_prompt | model | output_parser

# ==========================================
# INTERCONNECTION: Complete Multi-Tier Chain
# ==========================================
full_pipeline = (
    # Step 1: Generate Theory Notes
    {"context_notes": theory_chain}
    
    # Step 2: Generate Questions in Parallel
    | RunnableParallel({
        "theory": lambda x: x["context_notes"],
        "mcqs": mcq_gen_chain,
        "exercises": exercise_gen_chain
    })
    
    # Step 3: Solve Generated Questions in Parallel (Sequential to Step 2)
    | RunnableParallel({
        "theory": lambda x: x["theory"],            # Pass theory through to final dict
        "mcqs": lambda x: x["mcqs"],                # Pass raw MCQs through to final dict
        "exercises": lambda x: x["exercises"],      # Pass raw exercises through to final dict
        
        # Extract respective outputs from previous dictionary and feed forward
        "mcq_solutions": lambda x: x["mcqs"] | mcq_solve_chain,
        "exercise_solutions": lambda x: x["exercises"] | exercise_solve_chain
    })
)
full_pipeline.get_graph().print_ascii()
# ==========================================
# DYNAMIC WHILE LOOP EXECUTION
# ==========================================
def main():
    print("=== Multi-Tier Sequential & Parallel SQL Generator ===")
    print("Type 'exit' or 'quit' to stop the program.\n")
    
    while True:
        user_input = input("Enter an SQL topic (e.g., Left Join, Group By, CTEs): ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting generator. Happy coding!")
            break
            
        print(f"\nProcessing multi-tier pipeline for '{user_input}'...")
        print(" [1/3] Generating underlying theory notes...")
        print(" [2/3] Branching evaluation material (MCQs & Problems)...")
        print(" [3/3] Parallel processing answer keys and raw SQL solution scripts...")
        print("Please wait...\n")
        
        try:
            # Execute entire chain with one invocation
            response = full_pipeline.invoke({"topic": user_input})
            
            # Print Questions and Materials
            print("=" * 60)
            print(f"📖 GENERATED THEORY & NOTES: {user_input.upper()}")
            print("=" * 60)
            print(response["theory"])
            print("\n" + "=" * 60)
            
            print(f"❓ CONTEXT-BOUND MULTIPLE CHOICE QUESTIONS")
            print("=" * 60)
            print(response["mcqs"])
            print("\n" + "=" * 60)
            
            print(f"💻 PRACTICAL EXERCISES")
            print("=" * 60)
            print(response["exercises"])
            print("\n" + "=" * 60)
            
            # Keep solutions tucked below so user can opt to read or save separately
            print(f"🎯 ANSWER KEY: MULTIPLE CHOICE QUESTIONS")
            print("=" * 60)
            print(response["mcq_solutions"])
            print("\n" + "=" * 60)
            
            print(f"🛠️ FUNCTIONAL CODE SOLUTIONS: PRACTICAL EXERCISES")
            print("=" * 60)
            print(response["exercise_solutions"])
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"An error occurred while processing: {e}\n")

if __name__ == "__main__":
    main()
