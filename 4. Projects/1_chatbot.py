from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano",  # Set your OpenAI API key here if not using environment variables
                   )
chatHistory = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    chatHistory.append(HumanMessage(content=user_input))

    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))
    print(f"Assistant: {result.content}")

print(chatHistory)