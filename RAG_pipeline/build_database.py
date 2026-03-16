import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Define the local folder where our database will live
DB_DIR = "./chroma_db"

def build_vector_database(pdf_path: str):
    print(f"📄 Loading document: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find {pdf_path}")
        return

    # 2. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages.")

    # 3. Chunk the text
    # We split into 1000-character chunks with a 200-character overlap.
    # Overlap ensures we don't accidentally cut a sentence or concept in half!
    print("🔪 Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} semantic chunks.")

    # 4. Initialize the Embedding Model
    # all-MiniLM-L6-v2 is a phenomenal, lightweight embedding model for local machines
    print("🧠 Initializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Build and save the Vector Database (ChromaDB)
    print("🗄️ Calculating vectors and building ChromaDB. This might take a minute...")
    
    # We create the DB from our chunks and tell it to persist to our local folder
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"🚀 Success! Vector database saved locally to {DB_DIR}")

if __name__ == "__main__":
    print("="*50)
    print("🏦 Financial Document RAG: Ingestion Engine")
    print("="*50)
    
    # We need a PDF to test this on!
    target_pdf = "10K-NVDA.pdf"
    build_vector_database(target_pdf)