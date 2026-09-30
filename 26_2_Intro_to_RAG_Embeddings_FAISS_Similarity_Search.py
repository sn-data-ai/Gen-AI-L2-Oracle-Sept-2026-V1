import ollama 
import numpy as np
import faiss

# Step - 1 : Create sample data in form of documents
documents = [
    "This is a sample document about LangChain. LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, memory handling, and integration with external data sources.",
    "LangChain supports various document loaders, including text files, JSON files, CSV files, and PDF files. These loaders allow developers to easily ingest and process different types of data for use in their applications.",
    "Embeddings are a crucial component of LangChain, enabling semantic search and retrieval of relevant information. By converting text into numerical vectors, embeddings capture the meaning of the text and facilitate similarity-based retrieval.",
    "Vector stores in LangChain provide efficient storage and indexing of embeddings, allowing for fast retrieval.",
    "I am Sandip.",
    "I am from Pune.",
    "I am working in Zensar Technologies.",
    "I am technology trainer and Data & AI enthusiast.",
    "Pune is the city of Education",
    "Sachin Tendulkar is the God of Cricket.",
    "Sachin Tendulkar is from Mumbai.",
    "Generative AI (Gen AI) is a type of artificial intelligence that creates brand-new content, such as text, images, video, audio, and code, in response to user prompts."
]

# Store raw embedding arrays in a python list
embeddings_list = []

print("Generating embeddings for the documents...")

for i, doc in enumerate(documents):
    response = ollama.embeddings(
        model="nomic-embed-text",  # 768 dimensions
        prompt=doc
    )
    # Extract the actual embedding vector list from the response dictionary
    embeddings_list.append(response['embedding'])
    print(f"Document {i+1} embedding generated")

# ----------------------------------------------------
# Step - 2 : Implement FAISS Similarity Search
# ----------------------------------------------------

# 1. Convert the list of embeddings into a 2D NumPy array with float32 type (required by FAISS)
data_embeddings = np.array(embeddings_list).astype('float32')

# 2. Define the dimension of the embeddings (nomic-embed-text is 768)
dimension = data_embeddings.shape[1]

# 3. Create a FAISS index using L2 (Euclidean) distance
index = faiss.IndexFlatL2(dimension)

# 4. Add the document embeddings to the FAISS index
index.add(data_embeddings)
print(f"\nSuccessfully indexed {index.ntotal} documents in FAISS.")

# 5. Define a search query and generate its embedding
query = input("\nEnter your search query: ").strip()
print(f"\nQuery: '{query}'")

query_response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=query
)
query_embedding = np.array([query_response['embedding']]).astype('float32')

# 6. Perform the search for the top 'k' most similar documents
k = 3  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# 7. Display the retrieved results
print("\nTop Retrieved Chunks:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"Rank {rank+1}: (Distance/Score: {dist:.4f})")
    print(f"Document: {documents[idx]}\n")
