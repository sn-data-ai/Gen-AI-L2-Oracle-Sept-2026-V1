import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modern LangChain Integrations
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load environment variables from the .env file
load_dotenv()

# Extract the API Key from your .env data
api_key = os.getenv("OLLAMA_API_KEY")
if not api_key:
    print("❌ Error: 'OLLAMA_API_KEY' was not found in your .env file.")
    exit()

# Clean up any accidental surrounding quotes from the string extraction
api_key = api_key.strip("'\"")

# Expose it globally to the local environment session
os.environ["OLLAMA_API_KEY"] = api_key

# 2. Setup paths relative to the current working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, "data")
file_path = os.path.join(data_dir, "sample.txt")

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)
print(f"📁 Target data folder: {data_dir}")

# 3. Check if the file exists before attempting to load
if not os.path.exists(file_path):
    print(f"❌ Error: 'sample.txt' not found in the 'data' directory.")
    print("👉 Please paste the file manually and run the script again.")
    exit()

# 4. Load content using TextLoader
print(f"📖 Loading content from local file: {file_path}...")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 5. Split the documents into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Split file content into {len(chunks)} text chunks.")

# 6. Initialize Embeddings 
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

# 7. Create Chroma Vector Store
print("🧠 Generating embeddings and building ChromaDB index...")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
print("✅ ChromaDB vector database successfully built!")

# 8. Integrate Ollama Cloud LLM via Local Proxy Routing
print("\n🔄 Routing cloud request through local Ollama service...")
llm = ChatOllama(
    base_url="http://localhost:11434",  # Point to your local Ollama daemon
    model="gpt-oss:120b-cloud",         # The local proxy routes this to the cloud
    temperature=0.0,                    # Set to 0.0 to make responses strictly factual and deterministic
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }
)

# 9. Setup the strict prompt guardrail
# Instructs the model to only use the provided context and fail gracefully otherwise
system_prompt = (
    "You are a strict question-answering assistant. Your task is to answer the user's question "
    "using ONLY the provided context below. Do not use your own baseline knowledge, pre-trained "
    "facts, or any external assumptions outside of this text block.\n\n"
    "If the answer to the question cannot be explicitly found in the context below, or if the "
    "question is completely unrelated to the context, you must respond EXACTLY with: \"I don't know\".\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    """Combines document chunks into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

# Setup the retriever tool configuration
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 10. Modern LCEL RAG Chain Pipeline
rag_chain = (
    {
        "context": retriever | format_docs, 
        "input": RunnablePassthrough()
    }
    | prompt 
    | llm 
    | StrOutputParser()
)

# 11. Interactive Chat Loop
print("\n🤖 Dynamic Guardrailed Chat Bot is ready! (Type 'exit' or 'quit' to end the session)\n" + "="*60)

while True:
    try:
        # Capture user input
        user_query = input("\n👤 You: ").strip()
        
        # Check for exit condition
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat session. Goodbye!")
            break
            
        if not user_query:
            continue
            
        print("🔍 Searching context and generating response...")
        
        # Print out the specific metadata being inspected for this query
        retrieved_docs = vector_store.similarity_search(user_query, k=2)
        print("📄 Sources referenced:")
        for idx, doc in enumerate(retrieved_docs):
            print(f"   [{idx + 1}] Metadata: {doc.metadata}")
            
        # Invoke the pipeline
        response = rag_chain.invoke(user_query)
        
        print(f"\n✨ Bot: {response}")
        print("-" * 40)
        
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
