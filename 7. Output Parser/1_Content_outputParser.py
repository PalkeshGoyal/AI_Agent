from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
# )
# model = ChatGoogleGenerativeAI()
model = ChatOpenAI(model="gpt-4.1-mini", max_completion_tokens=150)

template1 = PromptTemplate(
    template="Write a detailed about the {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n{text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic":"AI"})
result = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result2 = model.invoke(prompt2)

print(result.content)
print(result2.content)

