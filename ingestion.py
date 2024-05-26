from dotenv import load_dotenv
import os
load_dotenv()

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def ingest_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")    
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(model="gemini-pro"), chain_type="stuff", retriever = new_vectorstore.as_retriever())

    print("****Loading to vectorstore done ***")
    return qa
