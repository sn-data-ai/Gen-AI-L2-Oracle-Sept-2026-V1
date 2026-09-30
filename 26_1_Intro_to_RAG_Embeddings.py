# Retrieval Augmented Generation (RAG) 
# External Data / Knowledge (Text, TextFile, JSON File, CSV, PDF files, etc.) 
        
# Document Loaders: Load external data into a structured format (e.g., text, JSON, CSV, PDF) for processing.
# Chunking: Break down large documents into smaller, manageable pieces (chunks) for efficient processing and retrieval.
# Embeddings: Convert text into numerical vectors that capture semantic meaning for similarity search and retrieval.
# Vector Stores: Store and index embeddings for efficient similarity search and retrieval of relevant document/chunks.
# LLMs: Use large language models to generate responses based on retrieved information and user queries.

# Step - 1 : Create sample data in form of documents ==>
            # External Knowledge 

documents = ["This is a sample document about LangChain. LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, memory handling, and integration with external data sources.",
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


import ollama 

# Store embeddings in python list variable.

documents_embeddings = []

print("Generating embeddings for the documents...")

for i, doc in enumerate(documents):
    response = ollama.embeddings(
        model="nomic-embed-text", # 768 dimensions
        prompt=doc
    )
    documents_embeddings.append(response)
    print(f"Document {i+1} embedding generated")