"""import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from dotenv import load_dotenv
#from langchain_community.llms import LlamaCpp
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import torch
load_dotenv()
# --- Verify API Key is loaded ---
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. "
                     "Please make sure you have a .env file in the same folder as this script, "
                     "and it contains GOOGLE_API_KEY='YOUR_KEY_HERE'")

# --- Path Configuration (for VS Code/Local Machine) ---
# 'data/' means a folder named 'data' in the same directory as this script
DATA_PATH = "data/"
# 'faiss_index/' means a folder named 'faiss_index' will be created/used here
FAISS_INDEX_DIR = "faiss_index/" # Changed to a directory for FAISS.save_local/load_local

# Ensure these directories exist. If not, they will be created.
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
print(f"Documents will be loaded from: {os.path.abspath(DATA_PATH)}")
print(f"FAISS index will be saved/loaded from: {os.path.abspath(FAISS_INDEX_DIR)}")
def load_all_documents(directory_path="data/"):
    all_documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
            all_documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF {filename}: {e}")
        elif filename.endswith(".docx"):
            try:
                loader = Docx2txtLoader(file_path)
                all_documents.extend(loader.load())
            except ImportError:
                print("Please install docx2txt: pip install docx2txt")
            except Exception as e:
                print(f"Error loading DOCX {filename}: {e}")
    return all_documents
data_directory = "data/"
all_loaded_documents = load_all_documents(data_directory)
print(f"Loaded {len(all_loaded_documents)} documents.")
splitter =  RecursiveCharacterTextSplitter( chunk_size=300, chunk_overlap=60 , separators=["\n\n", "\n", " ", ""])
documents = splitter.split_documents(all_loaded_documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#print("Creating/Updating FAISS index...")
db = FAISS.from_documents(documents , embeddings)
db.save_local("faiss.index")
print("FAISS index created/updated successfully!")
try:
    db = FAISS.load_local("faiss.index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully!")
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Retriever created successfully!")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    print(f"Error details: {e}")
#model_name = "microsoft/phi-2"
'''model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32 # Essential for optimal CPU performance and avoiding memory issues
)
llm = HuggingFacePipeline(pipeline=pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    device=-1 # CPU use karega
))'''
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, top_p=0.8)
print("LLM (Gemini 1.5 Flash) loaded via API.")
# --- CUSTOM PROMPT TEMPLATE FOR RAG ---
# This prompt guides the LLM to provide specific answers based ONLY on the context
template = You are a highly professional, empathetic, and knowledgeable First Aid Assistant. 
Your primary goal is to provide accurate, concise, and actionable first aid instructions based ONLY on the context provided.

**Instructions for Generating First Aid Advice:**
1. **Extract All Relevant Steps:** Identify and extract all available, relevant, and detailed step-by-step first aid instructions from the provided context.
2. **Specific & Non-Generic:** Ensure the answer is highly specific to the user's query and the context. Do NOT provide generic advice. Give techniques, exercises, and tactics professionally, not generically.
3. **Actionable & Flowing:** Present the instructions clearly and logically, as a natural, flowing numbered list. Each step should be a new numbered point.
4. **Self-Care First, Then Medical Help:** First, list all the steps the person can do on their own based on the context. AFTER all self-care steps are provided, always include a final numbered step advising to **"Seek professional medical assistance immediately."**
5. **No Contextual References:** Do NOT mention "provided text," "context," "information," or similar phrases in your answer. Just give the first aid advice directly.
6. **Give 4 to 5 points minimum as instructions.**

**Instructions for Handling Irrelevant/Out-of-Scope Queries:**
* If the user's question is NOT directly related to first aid (e.g., about major diseases like cancer, general medical advice, financial advice, or personal opinions), do NOT provide an answer. Instead, respond with: **"Sorry! I am designed for first aid assistance only."**




Context: {context}

Question: {question}

Helpful Answer:
1. 

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# --- RETRIEVAL QA CHAIN WITH CUSTOM PROMPT AND SOURCE DOCUMENTS ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True, # Set to True to see which documents were retrieved for debugging
    chain_type="stuff", # "stuff" chain type puts all retrieved docs into one context for LLM
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Apply the custom prompt HERE
)
# --- QUERIES ---
print("\n" + "="*80 + "\n")

query1 = "My friend slipped on ice and can't move their leg, and it looks swollen. What first aid should I provide until help arrives?"
print(f"Query 1: {query1}")
result1 = qa_chain({"query": query1}) # Use dict input for return_source_documents
answer1 = result1["result"]
retrieved_docs_for_query1 = result1["source_documents"] # Access source documents

print("💡 Answer (Query 1):", answer1)
print("\nRetrieved Documents (Query 1):")
for i, doc in enumerate(retrieved_docs_for_query1):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print("-" * 50)

print("\n" + "="*80 + "\n") # Separator

query2 = "Can you tell me how to build a nuclear reactor?"
print(f"Query 2: {query2}")
result2 = qa_chain({"query": query2}) # Use dict input for return_source_documents
answer2 = result2["result"]
retrieved_docs_for_query2 = result2["source_documents"] # Access source documents

print("💡 Answer (Query 2):", answer2)
print("\nRetrieved Documents (Query 2):")
for i, doc in enumerate(retrieved_docs_for_query2):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print("-" * 50)

print("\n" + "="*80 + "\n") # Separator
query3 = "How do I heal a broken arm?"
print(f"Query 3: {query3}")
result3 = qa_chain({"query": query3})
answer3 = result3["result"]
retrieved_docs_for_query3 = result3["source_documents"]

print("💡 Answer (Query 3):", answer3)
print("\nRetrieved Documents (Query 3):")
for i, doc in enumerate(retrieved_docs_for_query3):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print("-" * 50)
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it before running.")

# --- FAISS Index Path ---
FAISS_INDEX_DIR = "faiss_index/"

# --- Load Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load FAISS index (already built by build_index.py) ---
print("📂 Loading FAISS index...")
db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})
print("✅ FAISS index loaded successfully!")

# --- Load LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, top_p=0.8)
print("✅ LLM (Gemini 2.5 Flash) loaded via API.")

# --- Custom Prompt ---
template = """You are a highly professional, empathetic, and knowledgeable First Aid Assistant.
Your primary goal is to provide accurate, concise, and actionable first aid instructions based ONLY on the context provided .

Instructions for Generating First Aid Advice:
1. If relevant context is available from the retrieved documents, generate a clear, actionable, step-by-step first aid answer based on that context.
2. If relevant context is not available or insufficient, generate a professional, accurate, and actionable first aid answer based on your own knowledge, ensuring it is specific, practical, and safe.
3. Always present instructions in a logical, numbered list.
4. Avoid phrases like "based on the provided context" or "according to the information"; give the answer directly and professionally.
5. Handle greetings (e.g., "Hi", "Assalamualaikum") briefly and politely.
6. Do not answer irrelevant or out-of-scope questions. If the question is not related to first aid, reply: **"Sorry! I am designed for first aid assistance only."**
7. Avoid generic advice; give specific, professional techniques and instructions.
8. Provide at least 5–6 actionable points, or more if needed for clarity.
9. Self-care first, then advise: **"Seek professional medical assistance immediately."**. Always ,keep this line last .
10. Do not give the answer in headings . Just give cleanest  steps in simple text without any asterics and headings .

Instructions for Handling Missing or Insufficient Context:
If the retrieved context is empty, irrelevant, or insufficient to answer the query, generate a professional, self-contained, and actionable first aid response based on your knowledge.


Context: {context}
Question: {question}

Helpful Answer:
1.
"""



QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# --- Retrieval QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# --- Demo Query ---
print("\n" + "="*80 + "\n")
query = "My friend slipped on ice and can't move their leg, and it looks swollen. What first aid should I provide until help arrives?"
print(f"❓ Query: {query}")
result = qa_chain({"query": query})
answer = result["result"]

print("\n💡 Answer:")
print(answer)

print("\n📄 Retrieved Documents:")
for i, doc in enumerate(result["source_documents"]):
    print(f"--- Document {i+1} ---")
    print(doc.page_content[:300])  # sirf pehle 300 chars dikhaye for readability
    print("-" * 50)

