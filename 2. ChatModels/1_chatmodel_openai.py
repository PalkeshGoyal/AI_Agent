from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(
    model="gpt-4"  # Set your OpenAI API key here if not using environment variables,
)
result = chat.invoke("What is the capital of India?")  # Example invocation
print(result)  # Output the result of the invocation