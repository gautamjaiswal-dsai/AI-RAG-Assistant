import re
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# ======================
# ENV LOAD
# ======================
load_dotenv()

# ======================
# EMBEDDINGS (MUST MATCH DB)
# ======================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ======================
# LOAD VECTOR DB
# ======================
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ======================
# RETRIEVER (GOOD SETUP)
# ======================
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# ======================
# LLM (Groq)
# ======================
llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct"   # SAFE & STABLE
)

# ======================
# PROMPT
# ======================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If answer is not in context, say:
"I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

print("🚀 RAG system ready")
print("Type 0 to exit\n")

# ======================
# CHAT LOOP
# ======================
while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    output = response.content

    print(f"\nAI: {output}\n")