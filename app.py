import streamlit as st
from auth_module import check_password
from data_management import data_management
from chat_module import chat

# Set default values for session state
if 'AZURE_API_KEY' not in st.session_state:
    st.session_state['AZURE_API_KEY'] = "your-azure-api-key"
if 'AZURE_ENDPOINT' not in st.session_state:
    st.session_state['AZURE_ENDPOINT'] = "https://your-resource-name.openai.azure.com/"
if 'GPT4_DEPLOYMENT' not in st.session_state:
    st.session_state['GPT4_DEPLOYMENT'] = "gpt-35-turbo"

# Initialize session state for chat history, file selections, CSV visibility, and database connections
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'context_files' not in st.session_state:
    st.session_state['context_files'] = {}

if 'confirmed_context_files' not in st.session_state:
    st.session_state['confirmed_context_files'] = []

if 'csv_visibility' not in st.session_state:
    st.session_state['csv_visibility'] = {}

if 'db_connection_string' not in st.session_state:
    st.session_state['db_connection_string'] = ""

if 'db_tables' not in st.session_state:
    st.session_state['db_tables'] = []

if check_password():
    st.sidebar.image("logoapp.png", use_column_width=True)  # Display the logo at the top of the sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Data", "Model", "Chat"])

    if page == "Data":
        st.title("Data Management")
        data_management()

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
        chat()
