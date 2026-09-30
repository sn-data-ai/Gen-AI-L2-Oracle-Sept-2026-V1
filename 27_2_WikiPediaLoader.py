import faiss
import numpy as np
import ollama
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Fetch content from Wikipedia using WikipediaLoader
query = "Sachin Tendulkar"
print(f"🌐 Loading content from Wikipedia for query: '{query}'...")

# Fetch top 2 matching articles
loader = WikipediaLoader(query=query, load_max_docs=2)
docs = loader.load()

# 2. Split the documents into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Split documents into {len(chunks)} text chunks.")

# 3. Initialize Ollama Embeddings (Make sure your Ollama server is running)
# Replace 'llama3' with your locally pulled model (e.g., 'nomic-embed-text' or 'gemma')
embeddings = ollama.embeddings(model="nomic-embed-text", prompt=doc)

# 4. Create FAISS Vector Store and index the chunks
print("🧠 Generating embeddings and building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ Vector database successfully built!")

# 5. Perform a Similarity Search
search_query = "What is machine learning?"
print(f"\n🔍 Performing similarity search for: '{search_query}'")

# Retrieve the top 3 most relevant chunks
results = vector_store.similarity_search(search_query, k=3)

# Display results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} (Source: {doc.metadata.get('title', 'Unknown')}) ---")
    print(doc.page_content)
