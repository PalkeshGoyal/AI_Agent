from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
)

class Person(BaseModel):
    name : str = Field(description="the name of the person")
    age : int = Field(description="the age of the adult person", gt=18, lt=80)
    city : str = Field(description="the city of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {place} kid. \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

chain_input = {"place": "American"}
result = chain.invoke(chain_input)
print(result)