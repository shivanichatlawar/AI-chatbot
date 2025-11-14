"""
PDF Processing Module
Handles PDF text extraction and vector store creation
"""

import os
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class PDFProcessor:
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF file"""
        print(f"Extracting text from {self.pdf_path}...")
        reader = PdfReader(self.pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        print(f"Extracted {len(text)} characters from PDF")
        return text
    
    def create_documents(self, text: str) -> List[Document]:
        """Split text into chunks and create Document objects"""
        print("Splitting text into chunks...")
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"Created {len(documents)} document chunks")
        return documents
    
    def create_vector_store(self, documents: List[Document], api_key: str = None) -> Chroma:
        """Create and persist vector store from documents"""
        print("Creating vector store...")
        
        # Initialize embeddings
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        return vector_store
    
    def load_vector_store(self, api_key: str = None) -> Chroma:
        """Load existing vector store"""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings()
        
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}...")
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
            return vector_store
        else:
            raise FileNotFoundError(f"Vector store not found at {self.persist_directory}")
    
    def process_pdf(self, api_key: str = None, force_reprocess: bool = False) -> Chroma:
        """Main method to process PDF and create/load vector store"""
        # Check if vector store already exists
        if not force_reprocess and os.path.exists(self.persist_directory):
            try:
                return self.load_vector_store(api_key)
            except Exception as e:
                print(f"Error loading vector store: {e}. Reprocessing PDF...")
        
        # Process PDF
        text = self.extract_text_from_pdf()
        documents = self.create_documents(text)
        vector_store = self.create_vector_store(documents, api_key)
        
        return vector_store

