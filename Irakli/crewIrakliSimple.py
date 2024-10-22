from langchain_community.llms import Ollama

# Define the model
model = Ollama(model="gemma:2b")

# The email content with grammar issues
email = "Hi, Cann you send the documents soont? I didn't got them yet and need it by tomorrow."

# Create a function to prompt the model for grammar correction
def correct_grammar(email_content):
    prompt = f"Correct the grammar of the following email: '{email_content}'"
    response = model(prompt)
    return response

# Call the function and display the output
print("Correcting grammar...")
output = correct_grammar(email)
print("Corrected Email:", output)
