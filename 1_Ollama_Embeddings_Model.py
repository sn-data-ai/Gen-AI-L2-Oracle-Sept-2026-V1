import ollama


prompt = "What is the capital of India?"

response = ollama.embed(
    model='nomic-embed-text', # this is an embeddings model and not an LLM. 
    input= prompt,
)
print("The response from embeddings model for the input text is below embeddings \n :")

print(response.embeddings) # generate & display the embeddings (list of numbers -> Vectors )

#print("\n The length of the embeddings vector is:", len(response.embeddings)) # display the length of the embeddings vector


from ollama import chat

response = chat(
    model='gpt-oss:20b-cloud',
    messages=[{'role': 'user', 'content': prompt}],
)
print("\nThe response from the LLM for the prompt is:\n")
print(response.message.content)