import numpy as np
import faiss
import ollama
from langchain_community.document_loaders import WebBaseLoader

# 1. Fetch content from the web using WebBaseLoader
# You can pass a single URL string or a list of URLs

url = "https://docs.ollama.com/quickstart"

#url = "https://www.geeksforgeeks.org/python/python-variables/"

print(f"🌐 Loading content from: {url}...")

loader = WebBaseLoader(url)
docs = loader.load()

# Split the loaded web document text by sentences or paragraphs 
# so we index chunks instead of the entire webpage as one block
raw_text = docs[0].page_content
documents = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 20]

print(f"📄 Loaded and split webpage into {len(documents)} text chunks.")

# 2. Generate embeddings with Ollama
embeddings_list = []
print("🧠 Generating embeddings...")
for doc in documents:
    response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
    embeddings_list.append(response["embedding"])

# FAISS requires data as float32 NumPy arrays
embeddings_array = np.array(embeddings_list).astype('float32')

# 3. Initialize the FAISS Index
# nomic-embed-text outputs vectors with exactly 768 dimensions
dimension = 768
index = faiss.IndexFlatL2(dimension)

# Add our document vectors to the index
index.add(embeddings_array)
print(f"✅ Successfully indexed {index.ntotal} text chunks into FAISS.\n")

# 4. Define and Embed a Test Query
query = input("\nEnter Your Query: \n")
print(f"🔍 User Query: '{query}'")

query_response = ollama.embeddings(model="nomic-embed-text", prompt=query)
query_embedding = np.array([query_response["embedding"]]).astype('float32')

# 5. Perform the Similarity Search
k = 2  # Top 'k' closest matches to retrieve
# Safe check in case webpage yielded fewer chunks than 'k'
k = min(k, len(documents)) 

distances, indices = index.search(query_embedding, k)

# 6. Display the Retrieved Context
print("\n--- FAISS Search Results ---")
for idx, (dist, document_idx) in enumerate(zip(distances[0], indices[0])):
    # FAISS returns -1 for indices if it cannot find enough matches
    if document_idx == -1:
        continue
    print(f"\n[Rank {idx+1}] (Distance score: {dist:.4f})")
    print(f"Matched text: {documents[document_idx]}")
