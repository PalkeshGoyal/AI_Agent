from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
)
result = chat.invoke("What is the capital of India?")  # Example invocation
print(result.content)  # Output the result of the invocation