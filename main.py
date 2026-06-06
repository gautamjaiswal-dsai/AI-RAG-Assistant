from dotenv import load_dotenv 
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import re

load_dotenv()

def clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# model = ChatNVIDIA(model = "meta/llama-3.1-70b-instruct")
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",)
model = ChatGroq(model="qwen/qwen3-32b")

response = model.invoke("Explain Retrieval-Augmented Generation (RAG) in AI with a simple example.")

clean_text = clean_response(response.content)

print("\nGroq Response:\n")
print(clean_text)
