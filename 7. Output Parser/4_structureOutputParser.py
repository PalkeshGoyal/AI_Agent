from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
)

schema = [
    ResponseSchema(name="name", description="the name of the person"),
    ResponseSchema(name="age", description="the age of the person"),
    ResponseSchema(name="city", description="the city of the person"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template="Give me the name, age and city of a fictional person. \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({})
print(result)
print(result["name"])
print(result["age"])
print(result["city"])

print(type(result))