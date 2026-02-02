from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",  # Set your Google API key here if not using environment variables
)

detailed_prompt = PromptTemplate(
    template="Generate detailed information for {topic}",
    input_variables=["topic"],
)

important_prompt = PromptTemplate(
    template="Extract five important points from the following text:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = detailed_prompt | model | parser | important_prompt | model | parser     # langchain expression for chaining

chain_input = {"topic": "Ethanol in petrol"}

result = chain.invoke(chain_input)
print(result)

print(chain.get_graph().print_ascii())