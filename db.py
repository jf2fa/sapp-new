import streamlit as st
import pandas as pd
import sqlalchemy

def db_management():
    st.subheader("Database Management")

    db_connection_string = st.text_input("Database Connection String", value=st.session_state['db_connection_string'])
    db_query = st.text_area("SQL Query", value=st.session_state['db_query'])

    if st.button("Connect to Database and Execute Query"):
        try:
            engine = sqlalchemy.create_engine(db_connection_string)
            with engine.connect() as connection:
                result = connection.execute(db_query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                st.session_state['db_data'] = df
                st.session_state['db_visibility'] = True
                st.success("Query executed successfully.")
        except Exception as e:
            st.error(f"Error connecting to database or executing query: {e}")

    if st.session_state['db_visibility']:
        st.write("### Query Result")
        st.dataframe(st.session_state['db_data'])
