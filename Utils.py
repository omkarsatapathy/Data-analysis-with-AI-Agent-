import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from typing import Optional, List, Dict, Any
from pathlib import Path
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document


class OpenAIHandler:
    """
    Handles interactions with the OpenAI API
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
    def set_api_key(self, api_key):
        self.api_key = api_key
        
    def get_response(self, prompt, system_message="", model="gpt-4"):
        """
        Get a response from OpenAI API
        """
        if not self.api_key:
            return "Error: OpenAI API key is not set. Please provide your API key in the settings."
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message if system_message else "You are an AI assistant that helps with data analysis. You can write Python code to analyze data and create visualizations."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", 
                headers=headers, 
                data=json.dumps(data),
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error connecting to OpenAI API: {str(e)}"
        except KeyError as e:
            return f"Error parsing OpenAI response: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def generate_analysis_code(self, user_prompt, file_info, file_preview=None):
        """
        Generate Python code for data analysis based on user prompt and file info
        """
        system_message = """
        You are an expert data analyst AI assistant. Your task is to generate Python code to analyze data based on the user's request.
        The code you generate will be executed in a Python environment with the following libraries available:
        - pandas (as pd)
        - numpy (as np)
        - matplotlib.pyplot (as plt)
        - seaborn (as sns)

        A DataFrame is already loaded as 'df'. The code you generate should:
        1. Be executable without errors
        2. Include minimal comments - only those necessary to understand key steps
        3. Generate clear and informative visualizations when appropriate
        4. Print only essential information and insights
        5. ALWAYS include plt.show() when creating visualizations
        6. Be as concise as possible while still accomplishing the task

        Format your response with only the Python code inside triple backticks:
        ```python
        # Your code here
        ```
        """
        
        prompt = f"""
        USER REQUEST: {user_prompt}
        
        FILE INFO:
        {file_info}
        
        FILE PREVIEW:
        {file_preview if file_preview else 'No preview available'}
        
        Based on this information, generate Python code to analyze the data as requested.
        The code should be clean, efficient, and focus on producing the requested visualization or analysis with minimal unnecessary output.
        """
        
        return self.get_response(prompt, system_message)


class FileHandler:
    """
    Handles file operations
    """
    @staticmethod
    def load_file(file_path):
        """
        Load file from given path and return content and type
        """
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}", "error"
                
            file_type = file_path.split('.')[-1].lower()
            
            if file_type == 'csv':
                content = pd.read_csv(file_path)
                return content, 'csv'
            elif file_type in ['xlsx', 'xls']:
                content = pd.read_excel(file_path)
                return content, 'excel'
            elif file_type == 'parquet':
                content = pd.read_parquet(file_path)
                return content, 'parquet'
            elif file_type in ['txt', 'log']:
                with open(file_path, 'r') as file:
                    content = file.read()
                return content, 'text'
            elif file_type == 'json':
                with open(file_path, 'r') as file:
                    content = json.load(file)
                return content, 'json'
            elif file_type == 'pdf':
                # Just return the path for PDF files
                return file_path, 'pdf'
            else:
                return f"Unsupported file type: {file_type}", 'error'
        except pd.errors.EmptyDataError:
            return "Error: The file is empty.", 'error'
        except pd.errors.ParserError:
            return "Error: The file could not be parsed correctly.", 'error'
        except Exception as e:
            return f"Error loading file: {str(e)}", 'error'
    
    @staticmethod
    def export_dataframe(df, file_path, file_format='csv'):
        """
        Export DataFrame to the specified file format
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Ensure the file has the correct extension based on format
            if file_format.lower() == 'csv':
                if not file_path.lower().endswith('.csv'):
                    file_path = f"{file_path}.csv"
                df.to_csv(file_path, index=False)
                
            elif file_format.lower() in ['excel', 'xlsx', 'xls']:
                # For Excel files, ensure it has .xlsx extension
                if not any(file_path.lower().endswith(ext) for ext in ['.xlsx', '.xls']):
                    file_path = f"{file_path}.xlsx"
                
                try:
                    # Try to use openpyxl engine for .xlsx files
                    df.to_excel(file_path, index=False, engine='openpyxl')
                except ImportError:
                    return "Error: openpyxl is required for Excel export. Install it with 'pip install openpyxl'", False
                    
            elif file_format.lower() == 'parquet':
                if not file_path.lower().endswith('.parquet'):
                    file_path = f"{file_path}.parquet"
                
                try:
                    df.to_parquet(file_path, index=False)
                except ImportError:
                    return "Error: pyarrow or fastparquet is required for Parquet export. Install with 'pip install pyarrow'", False
            else:
                return f"Unsupported export format: {file_format}", False
                
            return f"File exported successfully to {file_path}", True
        except Exception as e:
            return f"Error exporting file: {str(e)}", False
    
    @staticmethod
    def get_file_info(df):
        """
        Get information about a DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            return "Not a DataFrame"
        
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        column_types = df.dtypes.to_string()
        
        missing_values = df.isnull().sum().to_string()
        
        preview = df.head(5).to_string()
        
        stats = ""
        try:
            stats = df.describe().to_string()
        except:
            stats = "Could not generate statistics"
        
        return {
            'info': info_str,
            'column_types': column_types,
            'missing_values': missing_values,
            'preview': preview,
            'stats': stats
        }
    
    @staticmethod
    def download_pdf_from_link(url, output_path=None, filename=None):
        """
        Download a PDF file from a URL and save it to local storage.
        
        Args:
            url (str): The URL of the PDF file
            output_path (str, optional): Directory to save the file. Defaults to current directory.
            filename (str, optional): Filename to save the PDF as. If None, extracts from URL.
            
        Returns:
            tuple: (filepath, success_message) if successful, (None, error_message) if failed
        """
        try:
            print(f"Starting download from URL: {url}")
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return None, "Invalid URL. Please provide a URL starting with http:// or https://"
            
            # Set default output directory to current directory if not specified
            if not output_path:
                output_path = os.getcwd()
            else:
                # Create directory if it doesn't exist
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
            
            # Set filename if not provided
            if not filename:
                # Extract filename from URL
                parsed_filename = os.path.basename(url.split('?')[0])
                if not parsed_filename or not parsed_filename.endswith('.pdf'):
                    # Generate a timestamp-based filename if extraction fails
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"downloaded_report_{timestamp}.pdf"
                else:
                    filename = parsed_filename
            
            # Ensure filename ends with .pdf
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            
            # Create full file path
            filepath = os.path.join(output_path, filename)
            
            print(f"Will save to: {filepath}")
            
            # Try to download the file with a lower timeout first to check if URL is valid
            try:
                # Initial HEAD request to validate URL and check content type
                head_response = requests.head(url, timeout=10)
                head_response.raise_for_status()
                print(f"HEAD request successful. Status code: {head_response.status_code}")
                print(f"Content-Type: {head_response.headers.get('Content-Type', 'unknown')}")
            except requests.exceptions.RequestException as e:
                print(f"HEAD request failed: {str(e)}")
                return None, f"Error validating PDF URL: {str(e)}"
            
            try:
                # Now do the actual download with streaming
                print("Starting file download...")
                response = requests.get(url, stream=True, timeout=60)  # Increased timeout for large PDFs
                response.raise_for_status()
                
                # Check if the content is likely a PDF
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                    print(f"Warning: Content-Type is not PDF: {content_type}")
                    # Proceed anyway but log the warning
                
                # Save the file with progress tracking
                file_size = int(response.headers.get('Content-Length', 0))
                print(f"File size: {file_size} bytes")
                
                # Save the file
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            downloaded += len(chunk)
                            print(f"Downloaded: {downloaded}/{file_size} bytes", end='\r')
                
                # Verify the downloaded file
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    print(f"File downloaded successfully: {filepath}")
                    print(f"File size: {os.path.getsize(filepath)} bytes")
                    return filepath, f"Successfully downloaded PDF to {filepath}"
                else:
                    return None, "Download completed but the file appears to be empty or corrupted."
                
            except requests.exceptions.Timeout:
                return None, "The download timed out. The server took too long to respond."
            except requests.exceptions.ConnectionError:
                return None, "Connection error. Please check your internet connection and the URL."
            except requests.exceptions.HTTPError as e:
                return None, f"HTTP Error: {e.response.status_code} - {e.response.reason}"
            except requests.exceptions.RequestException as e:
                return None, f"Error downloading PDF: {str(e)}"
                
        except Exception as e:
            import traceback
            print(f"Unexpected error in download_pdf_from_link: {str(e)}")
            print(traceback.format_exc())
            return None, f"Unexpected error: {str(e)}"


class CodeExecutor:
    """
    Handles Python code execution
    """
    @staticmethod
    def execute_code(code, df=None):
        """
        Execute Python code and capture output and figures
        """
        # Create local variables dictionary
        local_vars = {}
        
        # Add common libraries
        local_vars['pd'] = pd
        local_vars['np'] = np
        local_vars['plt'] = plt
        local_vars['sns'] = sns
        
        # Add DataFrame if available
        if df is not None and isinstance(df, pd.DataFrame):
            local_vars['df'] = df
        
        # Capture stdout and stderr
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            # Clear any existing figures
            plt.close('all')
            
            # Execute the code
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, globals(), local_vars)
            
            # Capture text output
            stdout_output = output_buffer.getvalue()
            stderr_output = error_buffer.getvalue()
            
            # Check if a figure was created
            figures = []
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    figures.append(fig)
            
            return {
                'output': stdout_output,
                'errors': stderr_output,
                'figures': figures,
                'local_vars': local_vars,
                'success': True
            }
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n\n{traceback.format_exc()}"
            return {
                'output': output_buffer.getvalue(),
                'errors': error_msg,
                'figures': [],
                'local_vars': local_vars,
                'success': False
            }


class PDFParser:
    """
    Handles PDF parsing and text extraction using only PyPDF2
    """
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file using PyPDF2
        
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
    def chunk_text(text, chunk_size=1000, chunk_overlap=200):
        """
        Split text into chunks for processing
        
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
    Retrieval-Augmented Generation system for answering questions about PDF documents
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.initialized = False
    
    def initialize_embeddings(self):
        """Initialize the OpenAI embeddings"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        return self.embeddings
    
    def create_vectorstore(self, documents):
        """
        Create a vector store from text documents
        
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
    
    def create_qa_chain(self):
        """
        Create a question-answering chain
        
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
    
    def process_pdf(self, pdf_path):
        """
        Process a PDF file and prepare it for question answering
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
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
    
    def answer_question(self, question):
        """
        Answer a question using the processed PDF
        
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


class AnnualReportAgent:
    """
    Enhanced implementation of an annual report retrieval agent using OpenAI API directly
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI handler
        self.openai_handler = OpenAIHandler(self.api_key)
    
    def find_and_download_annual_report(self, company_name: str) -> Optional[str]:
        """
        Find and download the latest annual report for a company using a more robust approach
        """
        try:
            # Step 1: Ask GPT for the investor relations URL for the company
            system_message = """
            You are a financial research expert. Your task is to provide the investor relations URL for the company.
            Respond with only the URL and nothing else. No explanations or additional text.
            """
            
            prompt = f"What is the URL for {company_name}'s investor relations website?"
            ir_url = self.openai_handler.get_response(prompt, system_message).strip()
            
            if not ir_url or not ir_url.startswith('http'):
                # Fallback to direct search format
                ir_url = f"https://www.{company_name.lower().replace(' ', '')}.com/investors"
            
            print(f"Found investor relations URL: {ir_url}")
            
            # Step 2: Now search for the annual report PDF link
            system_message_report = """
            You are a financial researcher. Based on the company's investor relations URL, provide the direct 
            download URL for their most recent annual report PDF.
            If you're uncertain about the exact URL, make your best guess for the most likely location 
            of the annual report on their investor site.
            Respond with only the URL and nothing else.
            """
            
            prompt_report = f"Based on this investor relations URL: {ir_url}, what is the direct URL for {company_name}'s most recent annual report PDF?"
            report_url = self.openai_handler.get_response(prompt_report, system_message_report).strip()
            
            print(f"Found potential annual report URL: {report_url}")
            
            # Try to download using the suggested URL
            if report_url and report_url.startswith('http'):
                # Manual download since PDFDownloadTool might not have enough context
                try:
                    response = requests.get(report_url, stream=True, timeout=30)
                    
                    # Check if we got a PDF
                    content_type = response.headers.get('Content-Type', '').lower()
                    if response.status_code == 200 and ('application/pdf' in content_type or report_url.lower().endswith('.pdf')):
                        # Create a clean filename
                        clean_name = re.sub(r'[^\w\s]', '', company_name).replace(' ', '_').lower()
                        # Extract year from URL if possible
                        year_match = re.search(r'20\d{2}', report_url)
                        year = year_match.group(0) if year_match else "latest"
                        
                        filename = f"{clean_name}_annual_report_{year}.pdf"
                        filepath = os.path.join(os.getcwd(), filename)
                        
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        print(f"Successfully downloaded report to: {filepath}")
                        return filepath
                except Exception as e:
                    print(f"Error downloading PDF: {e}")
            
            # If we got here, we need to try another approach - let's search with a more specific prompt
            system_message_specific = """
            You are a financial data researcher trying to find an annual report PDF.
            Based on your knowledge of corporate websites and investor relations pages, 
            provide the exact, direct download URL for the most recent annual report PDF for the company.
            Consider these possible URL patterns:
            1. https://www.company.com/investors/annual-reports/annual-report-2023.pdf
            2. https://investors.company.com/financial-information/annual-reports/default.aspx
            3. https://company.com/sites/default/files/annual_report_2023.pdf
            
            Respond with only the URL for the PDF download and nothing else.
            """
            
            prompt_specific = f"What is the direct download URL for {company_name}'s most recent annual report PDF file? I need the exact PDF file URL, not a page containing the link."
            specific_url = self.openai_handler.get_response(prompt_specific, system_message_specific).strip()
            
            print(f"Found alternate annual report URL: {specific_url}")
            
            # Try to download using the specific URL
            if specific_url and specific_url.startswith('http'):
                try:
                    response = requests.get(specific_url, stream=True, timeout=30)
                    if response.status_code == 200:
                        clean_name = re.sub(r'[^\w\s]', '', company_name).replace(' ', '_').lower()
                        year_match = re.search(r'20\d{2}', specific_url)
                        year = year_match.group(0) if year_match else "latest"
                        
                        filename = f"{clean_name}_annual_report_{year}.pdf"
                        filepath = os.path.join(os.getcwd(), filename)
                        
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        print(f"Successfully downloaded report to: {filepath}")
                        return filepath
                except Exception as e:
                    print(f"Error downloading PDF from specific URL: {e}")
            
            # If all attempts failed, create a dummy file with instructions
            # This is just to demonstrate the flow - in production you'd want to handle this differently
            dummy_filepath = os.path.join(os.getcwd(), f"{company_name.lower().replace(' ', '_')}_annual_report_info.txt")
            with open(dummy_filepath, 'w') as f:
                f.write(f"Please visit {ir_url} to access {company_name}'s annual reports.")
            
            return dummy_filepath
            
        except Exception as e:
            print(f"Error in annual report agent: {e}")
            return f"Error: {str(e)}"