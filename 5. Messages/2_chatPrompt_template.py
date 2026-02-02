from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate

chatTemplate = ChatPromptTemplate(
    [
        ("system","You are a helpful {domain} expert"),
        ("human","Explain the {topic} in simple words withing 5 lines")
    ]
)

prompt = chatTemplate.invoke({"domain":"Python", "topic":"decorators"})    
print(prompt)