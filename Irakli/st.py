import streamlit as st
from langchain_community.llms import Ollama

# Initialize the Ollama model
model = Ollama(model="gemma:2b")

# Function to correct grammar using the model
def correct_grammar(email_content):
    prompt = f"Correct the grammar of the following email: '{email_content}'"
    response = model(prompt)
    return response

# Streamlit app layout
st.title("Email Grammar Correction App")

# Modern Sidebar with buttons and options
st.sidebar.title("Menu")

# Add modern buttons to the sidebar
action = st.sidebar.radio(
    "Choose an Action:",
    ("Grammar Correction", "Sent Emails", "Settings")
)

st.sidebar.markdown("---")  # Horizontal line for separation

# Add an "About" section with a collapsible expander
with st.sidebar.expander("About this App"):
    st.write("""
    This app uses an AI language model to correct grammar mistakes in emails.
    Select "Grammar Correction" to correct an email, or explore other features in the menu.
    """)

# Add buttons for more modern interaction
st.sidebar.markdown("### Quick Actions")
if st.sidebar.button("Clear Email Input"):
    email_input = ""
else:
    email_input = st.text_area(
        "Enter your email:", 
        value="Hi, Can you send the documents soon? I didn't got them yet and need it by tomorrow."
    )

if st.sidebar.button("Submit Email"):
    if email_input:
        corrected_email = correct_grammar(email_input)
        st.write("### Corrected Email:")
        st.write(corrected_email)
    else:
        st.write("Please enter some email content.")

# Additional options in the sidebar
st.sidebar.markdown("### Settings")
grammar_style = st.sidebar.radio(
    "Select Grammar Style:",
    ("Formal", "Casual")
)

# Footer or additional information
st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ using Streamlit")

# Main area for email input/output
if action == "Grammar Correction":
    st.write("### Enter an email to correct its grammar")
elif action == "Sent Emails":
    st.write("### Sent Emails History (Not Implemented)")
elif action == "Settings":
    st.write("### App Settings (Not Implemented)")
