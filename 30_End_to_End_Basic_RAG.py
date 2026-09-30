import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
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

api_key = api_key.strip("'\"")
os.environ["OLLAMA_API_KEY"] = api_key

# 2. Setup paths relative to the current working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, "data")
txt_path = os.path.join(data_dir, "sample.txt")
pdf_path = os.path.join(data_dir, "gen-ai.pdf")

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)
print(f"📁 Target data folder: {data_dir}")

# 3. Dynamic loading of multiple file formats
docs = []

# Load text file if it exists
if os.path.exists(txt_path):
    print(f"📖 Loading text content: {txt_path}...")
    txt_loader = TextLoader(txt_path, encoding="utf-8")
    docs.extend(txt_loader.load())
else:
    print(f"⚠️ Warning: 'sample.txt' not found at {txt_path}")

# Load PDF file if it exists
if os.path.exists(pdf_path):
    print(f"📕 Loading PDF content: {pdf_path}...")
    pdf_loader = PyPDFLoader(pdf_path)
    docs.extend(pdf_loader.load())
else:
    print(f"⚠️ Warning: 'gen-ai.pdf' not found at {pdf_path}")

# Halt execution if no knowledge files are available
if not docs:
    print("❌ Error: Neither 'sample.txt' nor 'gen-ai.pdf' were found.")
    exit()

# 4. Split the combined documents into smaller text chunks
# Lowering chunk size slightly to get more granular matching chunks from the PDF
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Split combined file content into {len(chunks)} text chunks.")

# 5. Initialize Embeddings 
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

# 6. Create Chroma Vector Store
print("🧠 Generating embeddings and building ChromaDB index...")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
print("✅ ChromaDB vector database successfully built!")

# 7. Integrate Ollama Cloud LLM via Local Proxy Routing
print("\n🔄 Routing cloud request through local Ollama service...")
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gpt-oss:120b-cloud",
    temperature=0.0,
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }
)

# 8. Setup the strict prompt guardrail
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

# 🚀 Fix 1: Increase k to 5 or 6 so that PDF matching pages are not choked out by sample.txt vector slots
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# 9. Modern LCEL RAG Chain Pipeline
rag_chain = (
    {
        "context": retriever | format_docs, 
        "input": RunnablePassthrough()
    }
    | prompt 
    | llm 
    | StrOutputParser()
)

# 10. Interactive Chat Loop
print("\n🤖 Multi-Format Guardrailed Chat Bot is ready! (Type 'exit' or 'quit' to end the session)\n" + "="*60)

while True:
    try:
        user_query = input("\n👤 You: ").strip()
        
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat session. Goodbye!")
            break
            
        if not user_query:
            continue
            
        print("🔍 Searching context and generating response...")
        
        # 🚀 Fix 2: Pull expanded context to explicitly track what chunk files are entering the pipe
        retrieved_docs = vector_store.similarity_search(user_query, k=6)
        print("📄 Sources referenced:")
        for idx, doc in enumerate(retrieved_docs):
            # Extract just the filename to keep logging clean
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page_info = f" (Page {doc.metadata.get('page') + 1})" if 'page' in doc.metadata else ""
            print(f"   [{idx + 1}] File: {source_file}{page_info}")
            
        # Invoke the pipeline
        response = rag_chain.invoke(user_query)
        
        print(f"\n✨ Bot: {response}")
        print("-" * 40)
        
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
