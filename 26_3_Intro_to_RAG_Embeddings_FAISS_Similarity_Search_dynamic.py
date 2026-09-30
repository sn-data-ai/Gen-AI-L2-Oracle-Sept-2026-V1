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
# Step - 2 : Build FAISS Index
# ----------------------------------------------------

# Convert list to NumPy float32 array
data_embeddings = np.array(embeddings_list).astype('float32')
dimension = data_embeddings.shape[1]

# Create and populate L2 FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(data_embeddings)
print(f"\nSuccessfully indexed {index.ntotal} documents in FAISS.")

# ----------------------------------------------------
# Step - 3 : Dynamic Query Loop
# ----------------------------------------------------
print("\n--- FAISS Similarity Search System Ready ---")
print("Type your query below. Enter 'exit' or 'quit' to stop.")

while True:
    # Accept dynamic query input from the user
    user_query = input("\nEnter your search query: ").strip()
    
    # Check for exit condition
    if user_query.lower() in ['exit', 'quit']:
        print("Exiting search loop. Goodbye!")
        break
        
    # Skip empty inputs
    if not user_query:
        print("Please enter a valid query.")
        continue
        
    try:
        # Generate embedding for the dynamic query
        query_response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=user_query
        )
        query_embedding = np.array([query_response['embedding']]).astype('float32')

        # Perform the search for top k matches
        k = 3 
        distances, indices = index.search(query_embedding, k)

        # Display results
        print(f"\nTop {k} matches for: '{user_query}'")
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"  Rank {rank+1}: [Distance: {dist:.4f}]")
            print(f"  Document: {documents[idx]}\n")
            
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")
