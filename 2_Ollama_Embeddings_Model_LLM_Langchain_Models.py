from langchain_ollama import ChatOllama, OllamaEmbeddings

#prompt = "What is the capital of India?"

prompt = input("Enter your prompt:\n")

# 1. Initialize and use the Embeddings Model
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Generate the embedding vector for the prompt
# Note: embed_query returns a single list of floats, unlike embed_documents which takes a list of texts
embedding_vector = embeddings_model.embed_query(prompt)

print("The response from embeddings model for the input text is below embeddings \n:")
print(embedding_vector)
print("\nThe length of the embeddings vector is:", len(embedding_vector))


# 2. Initialize and use the Chat LLM
llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)

#llm = ChatOllama(model="gpt-5.6-sol", temperature=0)

# Invoke the LLM with the prompt
response = llm.invoke(prompt)

print("\nThe response from the LLM for the prompt is:\n")
# LangChain returns a AIMessage object; .content extracts the string response
print(response.content)

