from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  
load_dotenv()

documents = [
    "delhi is the capital of india",
    "mumbai is the financial capital of india",
    "kolkata is the cultural capital of india",
]

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Set your OpenAI API key here if not using environment variables
    dimensions=32,  # Specify the dimensions of the embeddings
)
embedding_results = embedding.embed_documents(documents)  # Example invocation
for i, result in enumerate(embedding_results):              
    print(f"Document {i+1} embedding: {str(result)}")  # Output the result of the embedding invocation
print(f"Total embeddings generated: {len(embedding_results)}")  # Output the total  