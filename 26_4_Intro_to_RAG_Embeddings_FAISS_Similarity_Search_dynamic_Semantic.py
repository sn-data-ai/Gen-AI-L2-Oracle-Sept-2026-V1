import ollama 
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

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

# ----------------------------------------------------
# Step - 2 : Setup BM25 (Keyword Search)
# ----------------------------------------------------
# Tokenize documents by lowercasing and splitting words
tokenized_documents = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# ----------------------------------------------------
# Step - 3 : Setup FAISS (Semantic Vector Search)
# ----------------------------------------------------
embeddings_list = []
print("Generating embeddings for FAISS Semantic Search...")

for i, doc in enumerate(documents):
    response = ollama.embeddings(
        model="nomic-embed-text",  # 768 dimensions
        prompt=doc
    )
    embeddings_list.append(response['embedding'])

data_embeddings = np.array(embeddings_list).astype('float32')
dimension = data_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(data_embeddings)
print(f"Successfully indexed {index.ntotal} documents in FAISS.")

# ----------------------------------------------------
# Step - 4 : Dynamic Query Loop (Side-by-Side Comparison)
# ----------------------------------------------------
print("\n--- Search Comparison System Ready ---")
print("Try queries like: 'Where does Sachin live?' or 'artificial intelligence'")
print("Type 'exit' or 'quit' to stop.")

while True:
    user_query = input("\nEnter your search query: ").strip()
    
    if user_query.lower() in ['exit', 'quit']:
        print("Exiting search loop. Goodbye!")
        break
        
    if not user_query:
        print("Please enter a valid query.")
        continue
        
    k = 2  # Top matches to fetch
    print("=" * 60)
    print(f"QUERY: '{user_query}'")
    print("=" * 60)

    # --- APPROACH A: KEYWORD SEARCH (BM25) ---
    tokenized_query = user_query.lower().split(" ")
    # Get similarity scores for all documents
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top k indices sorted by highest score
    bm25_indices = np.argsort(bm25_scores)[::-1][:k]
    
    print("\n[METHOD 1] SEMANTIC SEARCH (BM25) RESULTS:")
    for rank, idx in enumerate(bm25_indices):
        score = bm25_scores[idx]
        print(f"  Rank {rank+1}: [BM25 Score: {score:.4f}]")
        print(f"  Document: {documents[idx]}\n")

    # --- APPROACH B: SEMANTIC SEARCH (FAISS Vector Space) ---
    try:
        query_response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=user_query
        )
        query_embedding = np.array([query_response['embedding']]).astype('float32')
        distances, indices = index.search(query_embedding, k)

        print("-" * 40)
        print("[METHOD 2] SIMILARITY VECTOR SEARCH (FAISS) RESULTS:")
        # Flatten FAISS outputs for easier looping
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"  Rank {rank+1}: [L2 Distance: {dist:.4f}] (Lower is better)")
            print(f"  Document: {documents[idx]}\n")
            
    except Exception as e:
        print(f"An error occurred during vector search: {e}")
