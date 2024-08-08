import streamlit as st
from openai import AzureOpenAI
import pandas as pd

def get_chat_response(contexts, query, client):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"User: {query}"}
        ]

        for context in contexts:
            messages.insert(1, {"role": "system", "content": f"Context from {context['file']}:\n{context['content']}"})

        response = client.chat_completions.create(
            model=st.session_state['GPT4_DEPLOYMENT'],
            messages=messages,
            max_tokens=150
        )
        response_text = response.choices[0].message['content']
        return response_text
    except Exception as e:
        st.error(f"Error getting chat response: {e}")
        return None

def mock_chat_response(contexts, query):
    context_files = ', '.join([context['file'] for context in contexts])
    return f"This is a mock response based on the context from files: {context_files} and query: {query}"

def chat():
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
                if file == 'Database Query':
                    content = st.session_state['db_data'].to_csv(index=False)
                    contexts.append({"file": "Database Query", "content": content})
                else:
                    df = pd.read_csv(file)
                    content = df.to_csv(index=False)
                    contexts.append({"file": file, "content": content})

            if st.session_state['AZURE_API_KEY'] == "your-azure-api-key":
                response_text = mock_chat_response(contexts, user_query)
            else:
                client = AzureOpenAI(
                    api_key=st.session_state['AZURE_API_KEY'],
                    api_version="2024-02-01",
                    azure_endpoint=st.session_state['AZURE_ENDPOINT']
                )
                response_text = get_chat_response(contexts, user_query, client)
            st.session_state['chat_history'].append(f"AI: {response_text}")

    # Display chat history
    st.write("### Chat History")
    for message in reversed(st.session_state['chat_history']):
        st.write(message)
