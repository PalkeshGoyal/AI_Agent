from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()   

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Set your OpenAI API key here if not using environment variables
    dimensions=32,  # Specify the dimensions of the embeddings
)
documents = [
    "delhi is the capital of india",
    "God of cricket is sachin tendulkar",
    "mumbai is the financial capital of india",
    "AI is the future of technology",
    "House are build with bricks and cement",
    "Research is the key to innovation"
]

querty = "What is the future of technology?"
embedding_results = embedding.embed_documents(documents)  # Example invocation
query_embedding = embedding.embed_query(querty)  # Example invocation       

cosine_similarity_results = cosine_similarity(
    [query_embedding],
    embedding_results
)

for i, result in enumerate(cosine_similarity_results[0]):
    print(f"Document {i+1} similarity: {result:.4f}")  # Output the cosine similarity for each document
print(f"Most similar document index: {np.argmax(cosine_similarity_results[0])}")  # Output the index of the most similar document
print(f"Most similar document: {documents[np.argmax(cosine_similarity_results[0])]}")  # Output the most similar document
