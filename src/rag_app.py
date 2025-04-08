import os
from typing import List, Dict, Tuple, Any, Optional
import PyPDF2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document


class PDFParser:
    """
    Handles PDF parsing and text extraction from PDF documents.
    
    This class provides methods to extract text content from PDF files,
    chunk the extracted text into manageable pieces for processing,
    and prepare the content for retrieval-based systems.
    """
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Use PyPDF2 for text extraction
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text: {str(e)}"
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text (str): The text to split
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            print(f"Error chunking text: {str(e)}")
            return []


class RAGSystem:
    """
    Retrieval-Augmented Generation system for answering questions about PDF documents.
    
    This class integrates text extraction, embedding generation, and retrieval-based
    question answering to provide accurate responses to queries about PDF content.
    It creates a vector store from document chunks and uses semantic search to find
    relevant passages for answering user questions.
    """
    def __init__(self, api_key: str):
        """
        Initialize the RAG system with the OpenAI API key.
        
        Args:
            api_key (str): OpenAI API key for embeddings and model access
        """
        self.api_key = api_key
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.initialized = False
    
    def initialize_embeddings(self) -> OpenAIEmbeddings:
        """
        Initialize the OpenAI embeddings.
        
        Returns:
            OpenAIEmbeddings: Initialized embeddings object
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        return self.embeddings
    
    def create_vectorstore(self, documents: List[Any]) -> FAISS:
        """
        Create a vector store from text documents.
        
        Args:
            documents (list): List of Document objects or text chunks
            
        Returns:
            FAISS: Vector store with embedded documents
        """
        if self.embeddings is None:
            self.initialize_embeddings()
        
        # Convert text chunks to Document objects if needed
        if documents and isinstance(documents[0], str):
            docs = [Document(page_content=chunk, metadata={}) for chunk in documents]
        else:
            docs = documents
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        return self.vectorstore
    
    def create_qa_chain(self) -> RetrievalQA:
        """
        Create a question-answering chain.
        
        Returns:
            RetrievalQA: QA chain for answering questions
        """
        if self.vectorstore is None:
            raise ValueError("Vector store must be created first")
        
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=self.api_key
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        self.initialized = True
        return self.qa_chain
    
    def process_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """
        Process a PDF file and prepare it for question answering.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            tuple: (success status (bool), message (str))
        """
        try:
            # Extract text
            text = PDFParser.extract_text_from_pdf(pdf_path)
            if not text or text.startswith("Error"):
                return False, "Failed to extract text from PDF"
            
            # Split into chunks
            chunks = PDFParser.chunk_text(text)
            if not chunks:
                return False, "Failed to split text into chunks"
            
            # Create vector store
            self.create_vectorstore(chunks)
            
            # Create QA chain
            self.create_qa_chain()
            
            return True, f"Processed PDF with {len(chunks)} chunks"
        
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the processed PDF.
        
        Args:
            question (str): The question to answer
            
        Returns:
            dict: Answer and source documents
        """
        if not self.initialized:
            return {"answer": "The PDF has not been processed yet.", "sources": []}
        
        try:
            # Get answer
            result = self.qa_chain({"query": question})
            
            # Extract source texts
            sources = []
            for doc in result.get("source_documents", []):
                source_text = doc.page_content
                # Truncate long source texts
                if len(source_text) > 300:
                    source_text = source_text[:300] + "..."
                sources.append(source_text)
            
            return {
                "answer": result["result"],
                "sources": sources
            }
        
        except Exception as e:
            return {
                "answer": f"Error answering question: {str(e)}",
                "sources": []
            }


class DocumentQASession:
    """
    Manages a document question-answering session.
    
    This class provides a higher-level interface for document Q&A,
    handling the setup, document processing, question answering, and
    session state maintenance for document-based interactions.
    """
    def __init__(self, api_key: str):
        """
        Initialize a document Q&A session.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        self.rag_system = RAGSystem(api_key)
        self.current_document_path = None
        self.is_processed = False
        self.document_name = None
    
    def load_document(self, document_path: str) -> Tuple[bool, str]:
        """
        Load and process a document for question answering.
        
        Args:
            document_path (str): Path to the document file
            
        Returns:
            tuple: (success status (bool), message (str))
        """
        try:
            self.current_document_path = document_path
            self.document_name = os.path.basename(document_path)
            
            # Process the document
            success, message = self.rag_system.process_pdf(document_path)
            self.is_processed = success
            
            return success, message
        except Exception as e:
            self.is_processed = False
            return False, f"Error loading document: {str(e)}"
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the loaded document.
        
        Args:
            question (str): The question to answer
            
        Returns:
            dict: Answer and source documents
        """
        if not self.is_processed:
            return {
                "answer": "No document has been processed. Please load a document first.",
                "sources": []
            }
        
        return self.rag_system.answer_question(question)
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.
        
        Returns:
            dict: Session information
        """
        return {
            "document_path": self.current_document_path,
            "document_name": self.document_name,
            "is_processed": self.is_processed
        }