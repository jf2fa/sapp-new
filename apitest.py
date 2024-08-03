import os
from azure.ai.openai import AzureOpenAI

# Set your Azure OpenAI API key and endpoints
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
GPT4_DEPLOYMENT = "gpt-4"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2023-03-15-preview",
    azure_endpoint=AZURE_ENDPOINT
)

def test_embedding_api(text):
    try:
        response = client.get_embeddings(deployment_id=EMBEDDING_DEPLOYMENT, input=[text])
        embedding = response.data[0].embedding
        print("Embedding API test successful!")
        print(f"Embedding: {embedding[:5]}...")  # Print the first 5 dimensions for brevity
    except Exception as e:
        print(f"Error testing Embedding API: {e}")

def test_chat_api(context, user_query):
    try:
        response = client.create_chat_completion(
            deployment_id=GPT4_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"User: {user_query}"}
            ],
            max_tokens=150
        )
        response_text = response.choices[0].message['content']
        print("Chat API test successful!")
        print(f"AI Response: {response_text}")
    except Exception as e:
        print(f"Error testing Chat API: {e}")

if __name__ == "__main__":
    # Test embedding API
    test_text = "This is a test sentence for generating embeddings."
    test_embedding_api(test_text)

    # Test chat API
    test_context = "This is the context for the conversation."
    test_query = "What is the capital of France?"
    test_chat_api(test_context, test_query)
