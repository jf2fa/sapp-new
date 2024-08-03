import streamlit as st
import pandas as pd
from openai import OpenAI
import pickle
import os
import torch

# Placeholder variables for API keys and endpoints
AZURE_API_KEY = "your-azure-api-key"
EMBEDDING_ENDPOINT = "your-embedding-endpoint"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
GPT4_ENDPOINT = "your-gpt4-endpoint"
GPT4_DEPLOYMENT = "gpt-4o"

# Initialize session state for API keys and endpoints
if 'AZURE_API_KEY' not in st.session_state:
    st.session_state['AZURE_API_KEY'] = AZURE_API_KEY
if 'EMBEDDING_ENDPOINT' not in st.session_state:
    st.session_state['EMBEDDING_ENDPOINT'] = EMBEDDING_ENDPOINT
if 'EMBEDDING_DEPLOYMENT' not in st.session_state:
    st.session_state['EMBEDDING_DEPLOYMENT'] = EMBEDDING_DEPLOYMENT
if 'GPT4_ENDPOINT' not in st.session_state:
    st.session_state['GPT4_ENDPOINT'] = GPT4_ENDPOINT
if 'GPT4_DEPLOYMENT' not in st.session_state:
    st.session_state['GPT4_DEPLOYMENT'] = GPT4_DEPLOYMENT

# Initialize OpenAI client
client = OpenAI(api_key=st.session_state['AZURE_API_KEY'])

# Load precomputed embeddings and metadata if available
if os.path.exists("precomputed_embeddings.pkl") and os.path.exists("precomputed_metadata.pkl"):
    with open("precomputed_embeddings.pkl", "rb") as f:
        precomputed_embeddings = torch.tensor(pickle.load(f))
    with open("precomputed_metadata.pkl", "rb") as f:
        precomputed_metadata = pickle.load(f)
else:
    precomputed_embeddings = None
    precomputed_metadata = None

# Preload CSV files
preloaded_files = [
    "synthetic_survey_responses_1.csv",
    "synthetic_survey_responses_2.csv",
    "synthetic_survey_responses_3.csv",
    "synthetic_survey_responses.csv"
]

uploaded_files = []

# Simple password-based authentication
def check_password():
    def password_entered():
        if st.session_state["password"] == "dema":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Remove password from state after checking
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# Function to load a CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to generate embeddings
def generate_embeddings(text_data):
    try:
        response = client.embeddings.create(
            input=text_data,
            model=st.session_state['EMBEDDING_DEPLOYMENT']
        )
        embeddings = [data['embedding'] for data in response['data']]
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# Function to get chat response
def get_chat_response(context, query, embeddings=None):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"User: {query}"}
        ]
        if embeddings:
            messages.insert(1, {"role": "system", "content": f"Embeddings: {embeddings}"})
        
        response = client.chat.completions.create(
            model=st.session_state['GPT4_DEPLOYMENT'],
            messages=messages,
            max_tokens=150
        )
        response_text = response['choices'][0]['message']['content']
        return response_text
    except Exception as e:
        st.error(f"Error getting chat response: {e}")
        return None

# Define mock embedding function for local testing
def generate_mock_embedding(data):
    return [[0.1] * 768]  # Example of an embedding with 768 dimensions

# Define mock chat response function for local testing
def mock_chat_response(context, query):
    return "This is a mock response based on the context and query."

# Initialize session state for chat history, file selections, and CSV visibility
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'selected_files' not in st.session_state:
    st.session_state['selected_files'] = {file: False for file in preloaded_files}

if 'csv_visibility' not in st.session_state:
    st.session_state['csv_visibility'] = {file: False for file in preloaded_files}

if 'generated_embeddings' not in st.session_state:
    st.session_state['generated_embeddings'] = False

if check_password():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Data", "Model", "Chat"])

    if page == "Data":
        st.title("Data Management")

        st.subheader("Preloaded Files")
        for file in preloaded_files:
            cols = st.columns([3, 1])
            st.session_state['selected_files'][file] = cols[0].checkbox(file, value=st.session_state['selected_files'][file])
            if cols[1].button(f"View", key=f"view_{file}"):
                st.session_state['csv_visibility'][file] = not st.session_state['csv_visibility'][file]
            if st.session_state['csv_visibility'][file]:
                st.write(f"Preview of {file}:")
                st.dataframe(load_csv(file))
        
        st.subheader("Upload New Files")
        new_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
        if new_files:
            for new_file in new_files:
                cols = st.columns([3, 1])
                uploaded_files.append(new_file)
                st.session_state['selected_files'][new_file.name] = cols[0].checkbox(new_file.name, value=True)
                if cols[1].button(f"View", key=f"view_{new_file.name}"):
                    st.session_state['csv_visibility'][new_file.name] = not st.session_state['csv_visibility'][new_file.name]
                if st.session_state['csv_visibility'][new_file.name]:
                    st.write(f"Preview of {new_file.name}:")
                    st.dataframe(load_csv(new_file))

        if st.button("Generate Embeddings"):
            selected_files = [file for file, selected in st.session_state['selected_files'].items() if selected]
            combined_data = pd.concat([load_csv(file) for file in selected_files])
            text_data = combined_data.to_csv(index=False)

            if st.session_state['AZURE_API_KEY'] == "your-azure-api-key":
                embeddings = generate_mock_embedding(text_data)
            else:
                embeddings = generate_embeddings(text_data)

            if embeddings is not None:
                with open("precomputed_embeddings.pkl", "wb") as f:
                    pickle.dump(embeddings, f)
                with open("precomputed_metadata.pkl", "wb") as f:
                    pickle.dump(selected_files, f)

                st.session_state['generated_embeddings'] = True
                st.write("Embeddings generated and saved successfully!")
                st.write(f"Number of embeddings: {len(embeddings)}")
                st.write(f"Files used for embeddings: {', '.join(selected_files)}")

    elif page == "Model":
        st.title("Model Configuration")

        with st.expander("Embedding Model Configuration"):
            st.subheader("OpenAI API Credentials for Embedding Model")
            embedding_api_key = st.text_input("Embedding API Key", type="password", value=st.session_state['AZURE_API_KEY'])
            embedding_endpoint = st.text_input("Embedding Endpoint", value=st.session_state['EMBEDDING_ENDPOINT'])
            embedding_deployment = st.text_input("Embedding Deployment Name", value=st.session_state['EMBEDDING_DEPLOYMENT'])

            if st.button("Update Embedding Credentials"):
                st.session_state['AZURE_API_KEY'] = embedding_api_key
                st.session_state['EMBEDDING_ENDPOINT'] = embedding_endpoint
                st.session_state['EMBEDDING_DEPLOYMENT'] = embedding_deployment
                st.success("Embedding credentials updated successfully!")

        with st.expander("Chat Model Configuration"):
            st.subheader("OpenAI API Credentials for Chat Model")
            gpt4_api_key = st.text_input("Chat API Key", type="password", value=st.session_state['AZURE_API_KEY'])
            gpt4_endpoint = st.text_input("Chat Endpoint", value=st.session_state['GPT4_ENDPOINT'])
            gpt4_deployment = st.text_input("Chat Deployment Name", value=st.session_state['GPT4_DEPLOYMENT'])

            if st.button("Update Chat Credentials"):
                st.session_state['AZURE_API_KEY'] = gpt4_api_key
                st.session_state['GPT4_ENDPOINT'] = gpt4_endpoint
                st.session_state['GPT4_DEPLOYMENT'] = gpt4_deployment
                st.success("Chat credentials updated successfully!")

elif page == "Chat":
    st.title("Chat with GPT-4")

    user_query = st.text_input("Enter your query:")
    if st.button("Send Query"):
        if user_query:
            st.session_state['chat_history'].append(f"User: {user_query}")
            context = "\n".join(st.session_state['chat_history'])
            selected_files = ', '.join(precomputed_metadata) if precomputed_metadata else "No files"

            if st.session_state['AZURE_API_KEY'] == "your-azure-api-key":
                response_text = mock_chat_response(context, user_query)
            else:
                response_text = get_chat_response(context, user_query, precomputed_embeddings)
            
            if response_text is not None:
                st.session_state['chat_history'].append(f"AI: {response_text}")
        
    # Display chat history
    st.write("### Chat History")
    for message in reversed(st.session_state['chat_history']):
        st.write(message)

