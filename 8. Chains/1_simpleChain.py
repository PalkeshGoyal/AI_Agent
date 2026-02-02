from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
)

prompt = PromptTemplate(
    template="Generate five interesting points for {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = prompt | model | parser     # langchain expression for chaining

chain_input = {"topic": "Location Delhi"}
result = chain.invoke(chain_input)
print(result)

print(chain.get_graph().print_ascii())