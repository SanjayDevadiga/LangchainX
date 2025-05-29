from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Simple text invocation
result = llm.invoke("What is LangChain.")
print(result.content)