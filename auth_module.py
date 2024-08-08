import streamlit as st

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
