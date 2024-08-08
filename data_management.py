import streamlit as st
import pandas as pd

preloaded_files = [
    "synthetic_survey_responses_1.csv",
    "synthetic_survey_responses_2.csv",
    "synthetic_survey_responses_3.csv",
    "synthetic_survey_responses.csv"
]

uploaded_files = []

def load_csv(file):
    return pd.read_csv(file)

def data_management():
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
