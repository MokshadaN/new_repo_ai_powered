from ollama import Client

client = Client()

response = client.embeddings(model='bge-m3', prompt="This is a test")
print(response['embedding'])
