import requests
import json

# Variables to test
embedding_api_key = 'your_embedding_api_key'
embedding_endpoint = 'your_embedding_endpoint'
embedding_deployment_name = 'your_embedding_deployment_name'

chat_api_key = 'your_chat_api_key'
chat_endpoint = 'your_chat_endpoint'
chat_deployment_name = 'your_chat_deployment_name'

def test_embedding_model(api_key, endpoint, deployment_name):
    # Correctly formatted URL for Azure OpenAI embeddings endpoint
    url = f"{endpoint}/openai/deployments/{deployment_name}/embeddings?api-version=2022-12-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    data = {
        "input": ["This is a test input for the embedding model."]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Embedding model test successful!")
        print("Response:", response.json())
        # Expected output (example):
        # {
        #   "data": [
        #     {
        #       "embedding": [0.0023, 0.0152, -0.0076, ...],
        #       "index": 0
        #     }
        #   ]
        # }
    else:
        print("Embedding model test failed.")
        print("Status code:", response.status_code)
        print("Response:", response.text)

def test_chat_model(api_key, endpoint, deployment_name):
    # Correctly formatted URL for Azure OpenAI chat completion endpoint
    url = f"{endpoint}/openai/deployments/{deployment_name}/completions?api-version=2022-12-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    data = {
        "prompt": "This is a test input for the chat model.",
        "max_tokens": 50
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Chat model test successful!")
        print("Response:", response.json())
        # Expected output (example):
        # {
        #   "id": "cmpl-1234567890",
        #   "object": "text_completion",
        #   "created": 1234567890,
        #   "model": "text-davinci-003",
        #   "choices": [
        #     {
        #       "text": "This is the response from the chat model.",
        #       "index": 0,
        #       "logprobs": null,
        #       "finish_reason": "stop"
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 10,
        #     "completion_tokens": 10,
        #     "total_tokens": 20
        #   }
        # }
    else:
        print("Chat model test failed.")
        print("Status code:", response.status_code)
        print("Response:", response.text)

# Test the embedding model
test_embedding_model(embedding_api_key, embedding_endpoint, embedding_deployment_name)

# Test the chat model
test_chat_model(chat_api_key, chat_endpoint, chat_deployment_name)
