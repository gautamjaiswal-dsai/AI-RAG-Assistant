import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ======================
# DELETE OLD DB
# ======================
shutil.rmtree("chroma_db", ignore_errors=True)

# ======================
# PDF PATH (FIX HERE)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_path = os.path.join(
    BASE_DIR,
    "document_loaders",
    "Artificial Intelligence, Machine Learning, and Deep Learning.pdf"
)

# check file exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

# ======================
# LOAD PDF
# ======================
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print("Pages loaded:", len(docs))

# ======================
# SPLIT TEXT
# ======================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# ======================
# EMBEDDINGS
# ======================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ======================
# VECTOR DB
# ======================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

vectorstore.persist()

print("✅ DB recreated successfully")