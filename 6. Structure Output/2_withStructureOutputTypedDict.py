from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

class Review(TypedDict):
    key_themes : Annotated[list[str], "themes discussed in review in a list"]
    summary: Annotated[str, "A short summary of the review within 50-70 words."]
    sentiment : Annotated[str, "positive", "negative", "neutral","mixed"]
    pros : Annotated[Optional[list[str]], "pros of the product"]
    cons : Annotated[Optional[list[str]], "cons of the product"]

model = ChatOpenAI(model="gpt-4.1-mini", max_completion_tokens=150)

# text = "The product arrived two days earlier than expected, which was a pleasant surprise. At first glance, the packaging looked professional and secure. However, upon opening it, I noticed a small scratch on the surface of the device. While the scratch doesn’t affect functionality, it was disappointing to see this on a brand-new item. Setting up the device was straightforward, and the instructions were clear and easy to follow. After using it for a few days, I can say the performance is outstanding—fast, responsive, and exactly what I needed. Overall, despite the minor flaw, I’m quite satisfied with my purchase."

text = "I recently bought the XYZ Noise-Cancelling Headphones and have been using them daily. The sound quality is fantastic, especially the bass, and the noise cancellation works well even in noisy environments like public transport. The battery life is also impressive — I get almost 30 hours on a full charge. However, the ear cups can get a bit uncomfortable after long periods of use, and the Bluetooth connection sometimes drops unexpectedly. Overall, I'm happy with the purchase, but there’s definitely room for improvement."
structured_model = model.with_structured_output(Review)

result = structured_model.invoke(text)

print(result)
