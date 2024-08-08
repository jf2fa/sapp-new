import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

def load_csv(file):
    return pd.read_csv(file)

def load_db_table(table_name, engine):
    return pd.read_sql_table(table_name, engine)

def data_management():
    # Initialize session state variables
    if 'csv_visibility' not in st.session_state:
        st.session_state['csv_visibility'] = {}
    if 'db_connection_string' not in st.session_state:
        st.session_state['db_connection_string'] = ""
    if 'db_tables' not in st.session_state:
        st.session_state['db_tables'] = []

    # Section for preloaded files
    with st.expander("Preloaded Files", expanded=False):
        st.subheader("Preloaded Files")
        preloaded_files = [
            "synthetic_survey_responses_1.csv",
            "synthetic_survey_responses_2.csv",
            "synthetic_survey_responses_3.csv",
            "synthetic_survey_responses.csv"
        ]

        for file in preloaded_files:
            cols = st.columns([3, 1])
            if file not in st.session_state['context_files']:
                st.session_state['context_files'][file] = False
            st.session_state['context_files'][file] = cols[0].checkbox(file, value=st.session_state['context_files'][file])
            if cols[1].button(f"View", key=f"view_{file}"):
                st.session_state['csv_visibility'][file] = not st.session_state['csv_visibility'].get(file, False)
            if st.session_state['csv_visibility'].get(file, False):
                st.write(f"Preview of {file}:")
                st.dataframe(load_csv(file))

    # Section for uploading new files
    with st.expander("Upload New Files", expanded=False):
        st.subheader("Upload New Files")
        new_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
        if new_files:
            for new_file in new_files:
                cols = st.columns([3, 1])
                if new_file.name not in st.session_state['context_files']:
                    st.session_state['context_files'][new_file.name] = False
                st.session_state['context_files'][new_file.name] = cols[0].checkbox(new_file.name, value=True)
                if cols[1].button(f"View", key=f"view_{new_file.name}"):
                    st.session_state['csv_visibility'][new_file.name] = not st.session_state['csv_visibility'].get(new_file.name, False)
                if st.session_state['csv_visibility'].get(new_file.name, False):
                    st.write(f"Preview of {new_file.name}:")
                    st.dataframe(load_csv(new_file))

    # Section for database connection and table selection
    with st.expander("Database Connection", expanded=False):
        st.subheader("Database Connection")
        db_host = st.text_input("DB Host", value="localhost")
        db_port = st.text_input("DB Port", value="5432")
        db_name = st.text_input("DB Name", value="testdb")
        db_user = st.text_input("DB User", value="testuser")
        db_password = st.text_input("DB Password", type="password", value="password")

        if st.button("Connect to Database"):
            st.session_state['db_connection_string'] = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            engine = create_engine(st.session_state['db_connection_string'])
            st.session_state['db_tables'] = []

        if st.session_state['db_connection_string']:
            st.subheader("Execute SQL to Find Tables")
            sql_query = st.text_area("SQL Query to Find Tables", "SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            if st.button("Execute Query"):
                try:
                    engine = create_engine(st.session_state['db_connection_string'])
                    with engine.connect() as conn:
                        tables = conn.execute(text(sql_query))
                        st.session_state['db_tables'] = [row[0] for row in tables]
                        st.success("Query executed successfully.")
                except Exception as e:
                    st.error(f"Error executing query: {e}")

            # Display database tables
            st.subheader("Database Tables")
            for table in st.session_state['db_tables']:
                cols = st.columns([3, 1])
                if table not in st.session_state['context_files']:
                    st.session_state['context_files'][table] = False
                st.session_state['context_files'][table] = cols[0].checkbox(table, value=st.session_state['context_files'][table])
                if cols[1].button(f"View", key=f"view_{table}"):
                    st.session_state['csv_visibility'][table] = not st.session_state['csv_visibility'].get(table, False)
                if st.session_state['csv_visibility'].get(table, False):
                    engine = create_engine(st.session_state['db_connection_string'])
                    st.write(f"Preview of {table}:")
                    st.dataframe(load_db_table(table, engine))

    if st.button("Confirm Context Files"):
        st.session_state['confirmed_context_files'] = [file for file, selected in st.session_state['context_files'].items() if selected]
        st.success("Context files confirmed.")
        st.write("### Confirmed Context Files")
        for file in st.session_state['confirmed_context_files']:
            st.write(file)
