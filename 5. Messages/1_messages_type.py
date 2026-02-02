from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo",  # Set your OpenAI API key here if not using environment variables
                   max_completion_tokens=100)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who is the king khan of bollywood"),
]

result = model.invoke(messages)
# print(result)
messages.append(AIMessage(content=result.content))

print(messages)