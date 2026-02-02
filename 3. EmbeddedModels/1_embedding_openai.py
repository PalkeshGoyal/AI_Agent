from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Set your OpenAI API key here if not using environment variables
    dimensions=32, # Specify the dimensions of the embeddings   
)

embedding_result = embedding.embed_query("What is the capital of India?")  # Example invocation
print(str(embedding_result))  # Output the result of the embedding invocation
