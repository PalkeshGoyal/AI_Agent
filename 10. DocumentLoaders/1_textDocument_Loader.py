from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader(file_path="/media/palkeshbsai/e_drive1/CampusX/GenAI using LangChain/10. DocumentLoaders/test.txt", encoding="utf-8")
docs = loader.load()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

template = PromptTemplate(
    template="Summarise the Joke : \n{joke}",
    input_variables=["joke"],
)

chain = template | model | parser

print(chain.invoke(docs[0].page_content))