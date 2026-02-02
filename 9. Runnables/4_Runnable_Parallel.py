from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

template1 = PromptTemplate(
    template="Write a tweet content about {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write a detailed linkedin post about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

parallelChain = RunnableParallel({
    "tweet": RunnableSequence(template1, model, parser),
    "linkedin_post": RunnableSequence(template2, model, parser)
})

results = parallelChain.invoke({"topic": "AI in Healthcare within 50-100 words"})
print(results)