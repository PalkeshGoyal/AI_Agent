# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.schema.runnable import RunnableBranch, RunnableLambda

# load_dotenv()
# model_gcp = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",  # Set your Google API key here if not using environment variables
# )

# class SentimentResponse(BaseModel):
#     sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback, either 'positive' or 'negative'")


# parser = StrOutputParser()
# SentimentParser = PydanticOutputParser(pydantic_object=SentimentResponse)

# prompt1 = PromptTemplate(
#     template="Classify the sentiment of the following feedback text into positive or negative: \n{feedback} \n{format_instructions}",
#     input_variables=["feedback"],
#     partial_variables={"format_instructions": SentimentParser.get_format_instructions()},
# )


# classifier_chain = prompt1 | model_gcp | SentimentParser

# # responsePrompt_positive = PromptTemplate(
# #     template="Generate a positive response to the following feedback: \n{feedback}",
# #     input_variables=["feedback"],
# # )


# # responsePrompt_negative = PromptTemplate(
# #     template="Generate a negative response to the following feedback: \n{feedback}",
# #     input_variables=["feedback"],
# # )
# responsePrompt = PromptTemplate(
#     template="Generate a {sentiment} response to the following feedback: \n{feedback}",
#     input_variables=["sentiment","feedback"],
# )

# branch_chain = RunnableBranch(
#     (lambda x: x.sentiment == "positive" , responsePrompt | model_gcp | parser),
#     (lambda x: x.sentiment == "negative" , responsePrompt | model_gcp | parser),
#     RunnableLambda(lambda x: "The sentiment is neither positive nor negative.")
# )

# chain = classifier_chain | branch_chain

# result = chain.invoke({"feedback": "The product quality is excellent and I am very satisfied with my purchase!"})
# print(result)

# Above code is the simple code not the advance and effective logical code.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableMap

load_dotenv()

model_gcp = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback, either 'positive' or 'negative'"
    )

SentimentParser = PydanticOutputParser(pydantic_object=SentimentResponse)

# Classifier Prompt
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative: \n{feedback} \n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": SentimentParser.get_format_instructions()},
)

# Base classifier (returns only sentiment)
base_classifier = prompt1 | model_gcp | SentimentParser

# ✅ Wrap the classifier to return both sentiment and original feedback
classifier_chain = RunnableLambda(
    lambda x: {
        "feedback": x["feedback"],
        "sentiment": base_classifier.invoke({"feedback": x["feedback"]}).sentiment
    }
)

# Response generation prompt
responsePrompt = PromptTemplate(
    template="Generate a {sentiment} response to the following feedback: \n{feedback}",
    input_variables=["sentiment", "feedback"],
)

# Branch chain: Select response path based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", responsePrompt | model_gcp | StrOutputParser()),
    (lambda x: x["sentiment"] == "negative", responsePrompt | model_gcp | StrOutputParser()),
    RunnableLambda(lambda x: "The sentiment is neither positive nor negative.")
)

# Final chain: classifier -> response generation
chain = classifier_chain | branch_chain

# Test run
result = chain.invoke({"feedback": "The product quality is excellent and I am very satisfied with my purchase!"})
print(result)
