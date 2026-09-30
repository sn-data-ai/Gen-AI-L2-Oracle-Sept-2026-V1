import os
import faiss
import ollama
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 💡 Crucial Import: Make sure you use the LangChain Ollama integration package
from langchain_ollama import OllamaEmbeddings

# 1. Setup paths relative to the current working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, "data")
file_path = os.path.join(data_dir, "sample.txt")

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)
print(f"📁 Target data folder: {data_dir}")

# 2. Check if the file exists before attempting to load
if not os.path.exists(file_path):
    print(f"❌ Error: 'sample.txt' not found in the 'data' directory.")
    print("👉 Please paste the file manually and run the script again.")
    exit()

# 3. Load content using TextLoader
print(f"📖 Loading content from local file: {file_path}...")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 4. Split the documents into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Split file content into {len(chunks)} text chunks.")

# 5. Correct Instantiation: Initialize the LangChain Ollama Embeddings class
# Recommendation: use "nomic-embed-text" if you have pulled it, otherwise keep your pulled model name
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

# 6. Create FAISS Vector Store and index the chunks
print("🧠 Generating embeddings and building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ Vector database successfully built!")

# 7. Perform a Similarity Search
search_query = input("\nEnter your query:\n")
print(f"\n🔍 Performing similarity search for: '{search_query}'")

# Retrieve the top 2 most relevant chunks
results = vector_store.similarity_search(search_query, k=2)

# Display results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} (Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}) ---")
    print(doc.page_content)
    print("\nBelow are the document details from where the text is fetched.\n")
    print(doc.metadata)
