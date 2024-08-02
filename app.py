import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
from concurrent.futures import ThreadPoolExecutor
import requests
import json

# Initialize model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

model = load_model()

# Load precomputed data if available
def load_precomputed_data():
    try:
        with open('precomputed_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        with open('precomputed_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Convert embeddings to torch tensors with the correct dtype
        embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
        
        return torch.stack(embeddings), metadata  # Correctly stack the embeddings
    except FileNotFoundError:
        return None, None

precomputed_embeddings, precomputed_metadata = load_precomputed_data()

# Initialize in-memory storage for embeddings and metadata
if 'embeddings' not in st.session_state:
    if precomputed_embeddings is not None:
        st.session_state.embeddings = precomputed_embeddings
        st.session_state.metadata = precomputed_metadata
    else:
        st.session_state.embeddings = torch.empty(0, dtype=torch.float32)
        st.session_state.metadata = []

# Function to encode rows in parallel
def encode_rows(rows):
    return model.encode(rows, convert_to_tensor=True)

# Function to rebuild context
def rebuild_context(files):
    embeddings = []
    metadata = []

    for file in files:
        df = pd.read_csv(file)
        rows = [row.to_json() for _, row in df.iterrows()]

        # Encode rows in batches
        with ThreadPoolExecutor() as executor:
            batch_size = 64
            batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
            results = list(executor.map(encode_rows, batches))

        for batch in results:
            embeddings.extend(batch)

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

# Function to call Azure OpenAI API
def call_azure_openai(prompt, api_key):
    url = "https://<your-openai-endpoint>/v1/engines/<your-model-deployment>/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        st.error(f"API call failed: {response.status_code} - {response.text}")
        return None

# Password protection
def password_protection():
    if 'password_entered' not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        with st.expander("Enter Password", expanded=True):
            pwd = st.text_input("Password", type="password")
            if pwd:
                if pwd == "dema2024":
                    st.session_state.password_entered = True
                else:
                    st.error("Incorrect password")
                    st.stop()

# Run password protection
password_protection()

# UI components
if st.session_state.password_entered:
    st.title("Semantic Search App")

    st.sidebar.header("Actions")
    uploaded_files = st.sidebar.file_uploader("Select Data from Database", accept_multiple_files=True, type=["csv"])

    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []

    if uploaded_files:
        for file in uploaded_files:
            if st.sidebar.checkbox(file.name, key=file.name):
                if file not in st.session_state.selected_files:
                    st.session_state.selected_files.append(file)
            else:
                if file in st.session_state.selected_files:
                    st.session_state.selected_files.remove(file)

    if st.sidebar.button("Rebuild Context"):
        if st.session_state.selected_files:
            with st.spinner("Rebuilding context..."):
                files = [file for file in st.session_state.selected_files]
                rebuild_context(files)
        else:
            st.sidebar.warning("Please select at least one data source.")

    # Azure OpenAI API key input
    api_key = st.sidebar.text_input("Enter Azure OpenAI API Key", type="password")
    st.session_state.api_key = api_key

    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                if st.session_state.embeddings.numel() > 0:  # Ensure there are embeddings to search
                    results = semantic_search(query)
                    st.write("Top results:")
                    file_details = [f"File: {result[0]['file']}, Row: {result[0]['row']}, Score: {result[1]}" for result in results]
                    for detail in file_details:
                        st.write(detail)
                    
                    # Append query and results to conversation history
                    st.session_state.conversation_history.append(f"User: {query}")
                    st.session_state.conversation_history.append(f"Results: {', '.join([result[0]['row'] for result in results])}")
                    
                    # Generating a prompt based on the conversation history
                    conversation = "\n".join(st.session_state.conversation_history)
                    file_info = "\n".join(file_details)
                    prompt = f"The following entries were retrieved based on the user's query:\n\n{file_info}\n\nUser's query: {query}\n\nBased on these entries, please provide a detailed response."

                    if api_key:
                        with st.spinner("Calling Azure OpenAI API..."):
                            response = call_azure_openai(prompt, api_key)
                            if response:
                                st.write("OpenAI API Response:")
                                st.write(response)
                                st.session_state.conversation_history.append(f"OpenAI: {response}")
                else:
                    st.warning("No embeddings found. Please rebuild context with selected data sources.")
        else:
            st.warning("Please enter a query.")

    # Load precomputed data button
    if st.sidebar.button("Load Precomputed Data"):
        precomputed_embeddings, precomputed_metadata = load_precomputed_data()
        if precomputed_embeddings is not None:
            st.session_state.embeddings = precomputed_embeddings
            st.session_state.metadata = precomputed_metadata
            st.success("Precomputed data loaded successfully.")
        else:
            st.error("Precomputed data not found.")
