from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vector = embeddings.embed_query("What are embeddings?")
print(vector[:5])