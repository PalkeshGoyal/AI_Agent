from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(
    model="gpt-3.5-turbo-instruct"  # Set your OpenAI API key here if not using environment variables
)

result = openai.invoke("What is the capital of India?")  # Example invocation
print(result)  # Output the result of the invocation