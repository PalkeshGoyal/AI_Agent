from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def word_count(text: str) -> int:
    return len(text.split())

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

joke_template = PromptTemplate(
    template="Write a Joke about {topic}",
    input_variables=["topic"],
)

lambda_runnable = RunnableLambda(word_count)
passthrough_runnable = RunnablePassthrough()

joke_chain = RunnableSequence(joke_template, model, parser)

parallel_chain = RunnableParallel({
    "joke_length": lambda_runnable,
    "joke_pass": passthrough_runnable
    })

final_chain = RunnableSequence(
    joke_chain,
    parallel_chain
)
results = final_chain.invoke({"topic": "Cricket"})
print(results)