import streamlit as st
import pandas as pd
from openai import AzureOpenAI
import os
import re

# Set your Azure OpenAI API key and endpoints
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT4_DEPLOYMENT = "gpt-35-turbo"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)

# Initialize session state for API keys and endpoints
if 'AZURE_API_KEY' not in st.session_state:
    st.session_state['AZURE_API_KEY'] = AZURE_API_KEY
if 'AZURE_ENDPOINT' not in st.session_state:
    st.session_state['AZURE_ENDPOINT'] = AZURE_ENDPOINT
if 'GPT4_DEPLOYMENT' not in st.session_state:
    st.session_state['GPT4_DEPLOYMENT'] = GPT4_DEPLOYMENT

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
        # First run, show input for password
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Function to load a CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to normalize text
def normalize_text(s):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

# Function to get chat response
def get_chat_response(contexts, query):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"User: {query}"}
        ]
        
        for context in contexts:
            messages.insert(1, {"role": "system", "content": f"Context from {context['file']}:\n{context['content']}"})
        
        response = client.chat.completions.create(
            model=st.session_state['GPT4_DEPLOYMENT'],
            messages=messages,
            max_tokens=150
        )
        response_text = response.choices[0].message['content']
        return response_text
    except Exception as e:
        st.error(f"Error getting chat response: {e}")
        return None

# Define mock chat response function for local testing
def mock_chat_response(contexts, query):
    context_files = ', '.join([context['file'] for context in contexts])
    return f"This is a mock response based on the context from files: {context_files} and query: {query}"

# Initialize session state for chat history, file selections, and CSV visibility
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'selected_files' not in st.session_state:
    st.session_state['selected_files'] = {file: False for file in preloaded_files}

if 'csv_visibility' not in st.session_state:
    st.session_state['csv_visibility'] = {file: False for file in preloaded_files}

if check_password():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Data", "Model", "Chat"])

    if page == "Data":
        st.title("Data Management")

        # Load and select CSV files
        st.subheader("Preloaded Files")
        for file in preloaded_files:
            cols = st.columns([3, 1])
            st.session_state['selected_files'][file] = cols[0].checkbox(file, value=st.session_state['selected_files'][file])
            if cols[1].button(f"View", key=f"view_{file}"):
                st.session_state['csv_visibility'][file] = not st.session_state['csv_visibility'][file]
            if st.session_state['csv_visibility'][file]:
                st.write(f"Preview of {file}:")
                st.dataframe(load_csv(file))
        
        # Upload new CSV files
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

        # Add to context
        if st.button("Add to Context"):
            st.session_state['context_files'] = [file for file, selected in st.session_state['selected_files'].items() if selected]
            st.success("Files added to context.")

    elif page == "Model":
        st.title("Model Configuration")

        with st.expander("Chat Model Configuration"):
            st.subheader("OpenAI API Credentials for Chat Model")
            gpt4_api_key = st.text_input("Chat API Key", type="password")
            gpt4_endpoint = st.text_input("Chat Endpoint")
            gpt4_deployment = st.text_input("Chat Deployment Name")

            if st.button("Update Chat Credentials"):
                if gpt4_api_key:
                    st.session_state['AZURE_API_KEY'] = gpt4_api_key
                if gpt4_endpoint:
                    st.session_state['AZURE_ENDPOINT'] = gpt4_endpoint
                if gpt4_deployment:
                    st.session_state['GPT4_DEPLOYMENT'] = gpt4_deployment
                st.success("Chat credentials updated successfully!")

    elif page == "Chat":
        st.title("Chat with GPT-4")

        # Chat interface
        user_query = st.text_input("Enter your query:")
        if st.button("Send Query"):
            if user_query:
                st.session_state['chat_history'].append(f"User: {user_query}")

                # Gather contents from context files
                context_files = st.session_state.get('context_files', [])
                contexts = []
                for file in context_files:
                    df = load_csv(file)
                    content = df.to_csv(index=False)
                    contexts.append({"file": file, "content": content})

                if st.session_state['AZURE_API_KEY'] == "your-azure-api-key":
                    response_text = mock_chat_response(contexts, user_query)
                else:
                    response_text = get_chat_response(contexts, user_query)
                st.session_state['chat_history'].append(f"AI: {response_text}")

        # Display chat history
        st.write("### Chat History")
        for message in reversed(st.session_state['chat_history']):
            st.write(message)
