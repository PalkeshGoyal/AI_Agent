from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from regex import template

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
joke_template = PromptTemplate(
    template="Write a Joke about {topic}",
    input_variables=["topic"],
)
parser = StrOutputParser()

joke_chain = RunnableSequence(joke_template, model, parser)

passthrough_chain = RunnablePassthrough()   

joke_explain_template = PromptTemplate(
    template="Explain the joke: {joke}",
    input_variables=["joke"],
)
joke_explain_chain = RunnableSequence(joke_explain_template, model, parser)

parallelChain = RunnableParallel(
    {"joke" : passthrough_chain,
    "explanation" : joke_explain_chain}
)

final_chain = RunnableSequence(
    joke_chain,
    parallelChain
)
results = final_chain.invoke({"topic": "Cricket"})

print(results)