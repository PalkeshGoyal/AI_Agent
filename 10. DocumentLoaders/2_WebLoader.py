from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

url = "https://www.amazon.in/Samsung-Galaxy-Smartphone-Graphite-Storage/dp/B0DHL7YT5S/ref=sr_1_3?crid=X701R7YUQWZ4&dib=eyJ2IjoiMSJ9.0BWpZ4ImHW1ZfRnJShi1zg9k8jfEGY05v401fzX7EykONexPzMV9GsfuJiDinUQZf9T-gXWtP2jdesdDQCRznWhW6Qf7QnKsi823rkHlNl3B8ej6ADnUMl90y3UrxwsyMo-2V0G5rsDp0EKyig-r7A_GK_HnrhQz5bR85gVUuDKYS-attqjYk-50gIALXM4ylR9tmYSxi41PjSeKOxjzEm1mZMyQZ-tNIHMtwU8Q1Wo.NaDXb6wE159Ihi9Kn9ZG1kBTozZwMsIq_mYhkUkHnkg&dib_tag=se&keywords=samsung+s24+fe+5g&nsdOptOutParam=true&qid=1759230460&sprefix=samsung+%2Caps%2C247&sr=8-3"

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
parser = StrOutputParser()

loader = WebBaseLoader(url)