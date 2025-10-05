# build_index.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

DATA_PATH = "data/"
FAISS_INDEX_DIR = "faiss_index/"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

def load_all_documents(directory_path=DATA_PATH):
    all_documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            all_documents.extend(TextLoader(file_path).load())
        elif filename.endswith(".pdf"):
            try:
                all_documents.extend(PyPDFLoader(file_path).load())
            except Exception as e:
                print(f"Error loading PDF {filename}: {e}")
        elif filename.endswith(".docx"):
            try:
                all_documents.extend(Docx2txtLoader(file_path).load())
            except Exception as e:
                print(f"Error loading DOCX {filename}: {e}")
    return all_documents

print("📂 Loading documents...")
all_loaded_documents = load_all_documents()
print(f"✅ Loaded {len(all_loaded_documents)} documents.")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
documents = splitter.split_documents(all_loaded_documents)
print(f"✅ Split into {len(documents)} chunks.")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(documents, embeddings)
db.save_local(FAISS_INDEX_DIR)

print(f"✅ FAISS index created and saved to {FAISS_INDEX_DIR}")
