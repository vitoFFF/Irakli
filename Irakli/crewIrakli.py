from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

# Define the model
model = Ollama(model="gemma:2b")

# The email content
email = "Hii, Can you sendt the documents soonn? I didn't got them yet and need it by tomorrow."

# Define the grammar correction agent
print("Defining grammar correction agent...")
grammar_corrector = Agent(
    role="grammar corrector",
    goal="Correct any grammar mistakes in the given email while preserving the original meaning.",
    backstory="You are an AI assistant whose job is to ensure all written text is grammatically correct while keeping the original message intact.",
    verbose=False,  # Set to True or False
    allow_delegation=False,
    max_iter=15,
    llm=model
)

# Create a task to correct the grammar of the email
print("Creating task to correct grammar...")
correct_grammar = Task(
    description=f"Correct the grammar mistakes in the following email: '{email}'",
    agent=grammar_corrector,
    expected_output="The grammatically correct version of the email.",
)

# Create the crew with the grammar corrector agent
crew = Crew(
    agents=[grammar_corrector],
    tasks=[correct_grammar],
    verbose=False,  # Set to True or False
    process=Process.sequential
)

# Start the process
print("Starting Crew process...")
output = crew.kickoff()
print("Output:", output)
