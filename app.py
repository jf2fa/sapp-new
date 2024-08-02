import streamlit as st
import pandas as pd
import torch
import requests
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from langchain_community.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set up page configuration
st.set_page_config(page_title="Semantic Search App", layout="wide")

# Initialize local embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# Function to call Azure OpenAI API for embeddings
def call_azure_embeddings_api(texts, api_key, endpoint, deployment_name):
    api_version = "2022-12-01"
    url = f"{endpoint}/openai/deployments/{deployment_name}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    response = requests.post(url, headers=headers, json={"input": texts})
    if response.status_code == 200:
        return response.json()["data"]
    st.error(f"API call failed: {response.status_code} - {response.text}")
    return None

# Function to call Azure OpenAI API for chat
def call_azure_chat_api(conversation, api_key, endpoint, deployment_name):
    api_version = "2022-12-01"
    url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    data = {
        "messages": [{"role": "system", "content": "You are an assistant."}, {"role": "user", "content": conversation}],
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    st.error(f"API call failed: {response.status_code} - {response.text}")
    return None

# Initialize session state variables
if 'embeddings' not in st.session_state:
    st.session_state.embeddings, st.session_state.metadata = torch.empty(0, dtype=torch.float32), []

# Load precomputed data if available
def load_precomputed_data():
    try:
        with open('precomputed_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        with open('precomputed_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        embeddings = [torch.tensor(e) for e in embeddings]
        return torch.stack(embeddings), metadata
    except FileNotFoundError:
        return torch.empty(0, dtype=torch.float32), []

# Load precomputed data into session state
if st.session_state.embeddings.numel() == 0:
    st.session_state.embeddings, st.session_state.metadata = load_precomputed_data()

# Function to encode rows using local model
def encode_rows_locally(rows):
    return model.encode(rows, convert_to_tensor=True)

# Function to encode rows using Azure model
def encode_rows_azure(rows, api_key, endpoint, deployment_name):
    results = call_azure_embeddings_api(rows, api_key, endpoint, deployment_name)
    if results:
        embeddings = [torch.tensor(result["embedding"]) for result in results]
        return torch.stack(embeddings) if embeddings else torch.empty(0, dtype=torch.float32)
    return torch.empty(0, dtype=torch.float32)

# Function to rebuild context
def rebuild_context(files, embedding_source, api_key, endpoint, deployment_name):
    embeddings = []
    metadata = []
    for file in files:
        df = pd.read_csv(file)
        rows = [row.to_json() for _, row in df.iterrows()]
        encoded_rows = encode_rows_locally(rows) if embedding_source == "Local" else encode_rows_azure(rows, api_key, endpoint, deployment_name)
        embeddings.extend(encoded_rows)
        metadata.extend([{'file': file.name, 'row': row} for row in rows])
    st.session_state.embeddings = torch.stack(embeddings) if embeddings else torch.empty(0, dtype=torch.float32)
    st.session_state.metadata = metadata
    st.info(f"Rebuilt vector database with {len(files)} files.")

# Function to perform semantic search
def semantic_search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, st.session_state.embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [(st.session_state.metadata[idx], score.item()) for idx, score in zip(top_results.indices, top_results.values)]

# UI components
if 'password_entered' not in st.session_state:
    st.session_state.password_entered = False

if not st.session_state.password_entered:
    st.header("Password")
    pwd = st.text_input("Enter Password", type="password")
    if st.button("Submit Password"):
        if pwd == "dema2024":
            st.session_state.password_entered = True
        else:
            st.error("Incorrect password")
else:
    tab1, tab2, tab3 = st.tabs(["Chat", "Data", "Model Settings"])

    # Chat Tab
    with tab1:
        st.header("Chat")
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        query = st.text_input("Enter your query:")

        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    if st.session_state.embeddings.numel() > 0:
                        results = semantic_search(query)
                        st.session_state.conversation_history.append(f"User: {query}")
                        
                        # Check if chat model credentials are provided
                        if st.session_state.chat_api_key and st.session_state.chat_endpoint and st.session_state.chat_deployment_name:
                            conversation = "\n".join(st.session_state.conversation_history)
                            prompt = f"The following are some relevant entries based on the user's query:\n\n{conversation}\n\nBased on these entries, please provide a detailed response to the query: {query}"
                            response = call_azure_chat_api(prompt, st.session_state.chat_api_key, st.session_state.chat_endpoint, st.session_state.chat_deployment_name)
                            if response:
                                st.session_state.conversation_history.append(f"OpenAI: {response}")
                                st.write(f"OpenAI: {response}")
                            else:
                                st.session_state.conversation_history.append("LangChain is working, but no response from the chat model due to missing or invalid credentials.")
                                st.write("LangChain is working, but no response from the chat model due to missing or invalid credentials.")
                        else:
                            st.write("Top results:")
                            for result in results:
                                st.write(f"File: {result[0]['file']}, Row: {result[0]['row']}, Score: {result[1]}")
                            st.session_state.conversation_history.append(f"Results: {', '.join([result[0]['row'] for result in results])}")
                    else:
                        st.warning("No embeddings found. Please rebuild context with selected data sources.")
            else:
                st.warning("Please enter a query.")

        # Display conversation history in reverse order (most recent first)
        for message in reversed(st.session_state.conversation_history):
            st.write(message)

    # Data Tab
    with tab2:
        st.header("Data")
        uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
        if 'selected_files' not in st.session_state:
            st.session_state.selected_files = []
        if uploaded_files:
            for file in uploaded_files:
                if st.checkbox(file.name, key=file.name):
                    if file not in st.session_state.selected_files:
                        st.session_state.selected_files.append(file)
                else:
                    if file in st.session_state.selected_files:
                        st.session_state.selected_files.remove(file)
        if st.button("Rebuild Context"):
            if st.session_state.selected_files:
                with st.spinner("Rebuilding context..."):
                    files = [file for file in st.session_state.selected_files]
                    rebuild_context(files, st.session_state.embedding_source, st.session_state.embedding_api_key, st.session_state.embedding_endpoint, st.session_state.embedding_deployment_name)
            else:
                st.warning("Please select at least one data source.")
        if st.button("Load Precomputed Data"):
            precomputed_embeddings, precomputed_metadata = load_precomputed_data()
            if precomputed_embeddings is not None:
                st.session_state.embeddings = precomputed_embeddings
                st.session_state.metadata = precomputed_metadata
                st.success("Precomputed data loaded successfully.")
            else:
                st.error("Precomputed data not found.")

# Model Settings Tab
    with tab3:
        st.header("Model Settings")
        
        st.subheader("Embedding Model Settings")
        embedding_api_key = st.text_input("Enter Azure OpenAI API Key for Embeddings", type="password")
        embedding_endpoint = st.text_input("Enter Azure OpenAI Endpoint URL for Embeddings")
        embedding_deployment_name = st.text_input("Enter Embedding Model Deployment Name")
        
        if st.button("Submit Embedding Model Settings"):
            st.session_state.embedding_api_key = embedding_api_key
            st.session_state.embedding_endpoint = embedding_endpoint
            st.session_state.embedding_deployment_name = embedding_deployment_name
            
            if st.session_state.embedding_api_key and st.session_state.embedding_endpoint and st.session_state.embedding_deployment_name:
                st.success(f"**Embedding Model Configured:**")
                st.write(f"API Key: {st.session_state.embedding_api_key}")
                st.write(f"Endpoint: {st.session_state.embedding_endpoint}")
                st.write(f"Deployment Name: {st.session_state.embedding_deployment_name}")

        st.subheader("Chat Model Settings")
        chat_api_key = st.text_input("Enter Azure OpenAI API Key for Chat", type="password")
        chat_endpoint = st.text_input("Enter Azure OpenAI Endpoint URL for Chat")
        chat_deployment_name = st.text_input("Enter Chat Model Deployment Name")
        
        if st.button("Submit Chat Model Settings"):
            st.session_state.chat_api_key = chat_api_key
            st.session_state.chat_endpoint = chat_endpoint
            st.session_state.chat_deployment_name = chat_deployment_name
            
            if st.session_state.chat_api_key and st.session_state.chat_endpoint and st.session_state.chat_deployment_name:
                st.success(f"**Chat Model Configured:**")
                st.write(f"API Key: {st.session_state.chat_api_key}")
                st.write(f"Endpoint: {st.session_state.chat_endpoint}")
                st.write(f"Deployment Name: {st.session_state.chat_deployment_name}")