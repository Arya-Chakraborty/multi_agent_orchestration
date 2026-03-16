import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Disable tokenizer parallelism warning from HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Configuration
DB_DIR = "./chroma_db"
print("🧠 Initializing models...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3.1", temperature=0)

# 2. Connect to the existing Vector Database
print("🗄️ Connecting to local Vector Database...")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 3. Create the Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

system_prompt = (
    "You are an elite Financial Analyst for BlackRock"
    "Use the following pieces of retrieved context from a company's financial report to answer the user's question. "
    "CRITICAL RULES: "
    "1. If the answer is not contained in the context, you MUST say 'I cannot answer this based on the provided document.' "
    "2. DO NOT use outside knowledge or hallucinate numbers. "
    "3. Be concise, professional, and directly answer the question.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 5. Build the LangChain RAG Pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🕵️  Financial RAG Interrogator Online (Local Llama 3.1)")
    print("="*50)
    
    while True:
        query = input("\nAsk a question about the PDF (or type 'quit'): ")
        if query.lower() in ['quit', 'exit']:
            break
            
        print("\n🔍 Scanning document vectors...\n")
        
        # Execute the chain
        response = rag_chain.invoke({"input": query})
        
        # Print the LLM's generated answer
        print("🤖 [ANALYSIS]:")
        print(response["answer"])
        
        # Print the EXACT sources it used to prove it isn't hallucinating
        print("\n" + "-"*40)
        print("📑 SOURCES USED:")
        for i, doc in enumerate(response["context"]):
            # Extract page number if the PDF loader captured it
            page = doc.metadata.get('page', 'Unknown Page')
            # Show a snippet of the text
            snippet = doc.page_content.replace('\n', ' ')[:150]
            print(f"  [{i+1}] Page {page}: {snippet}...")