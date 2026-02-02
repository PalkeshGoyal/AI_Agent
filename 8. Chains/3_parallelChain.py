from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel

load_dotenv()
model_gcp = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
)

model_openai = ChatOpenAI(model="gpt-4.1-mini",
                           # max_completion_tokens=150
                           )

template1 = PromptTemplate(
    template="Generate short and simple notes from the following text: \n{text}",
    input_variables=["text"],
)

template2 = PromptTemplate(
    template="Generate 5 short question and answers from the following text: \n{text}",
    input_variables=["text"],
)

template3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document: \nNotes: {notes}\nQuiz: {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": template1 | model_gcp | parser,
    "quiz": template2 | model_openai | parser
})

merge_chain = template3 | model_gcp | parser

chain = parallel_chain | merge_chain

text = """OpenAI is an artificial intelligence research and deployment company founded in December 2015 with the mission to ensure that artificial general intelligence (AGI) benefits all of humanity. Originally established as a non-profit, OpenAI later transitioned to a capped-profit model to attract funding while maintaining its commitment to ethical AI development. The organization is known for creating advanced AI models such as GPT (Generative Pre-trained Transformer), including ChatGPT, which can understand and generate human-like language. OpenAI focuses on safety, transparency, and long-term research goals, and it collaborates with global partners to guide the responsible development and use of AI technologies. Its tools are used in education, research, business, and creative fields, highlighting AI's potential across diverse domains.
"""

result = chain.invoke({"text": text})
print(result)
print(chain.get_graph().print_ascii())