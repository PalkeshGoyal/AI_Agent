from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

# Loading environment variables stored in .env file
load_dotenv()

# Creating the object of Google Gemini model
model = ChatGoogleGenerativeAI(
    # model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
    # model = "gemini-pro"
    model="gemini-2.5-flash"
)

# model = ChatOpenAI(model="gpt-4.1-mini", max_completion_tokens=150)

# Created a simple template
template1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Explain the Joke in detail {text}",
    input_variables=["text"],
)

# Created the output parser object
parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)

print(chain.invoke({"topic": "Cricket"}))
