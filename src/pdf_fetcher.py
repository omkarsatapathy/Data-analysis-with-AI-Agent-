import os
import re
import requests
from typing import Tuple, Optional, Dict, Any
import traceback
from src.llm import OpenAIHandler


class PDFDownloader:
    """
    Handles downloading PDF files from URLs.
    
    This class provides methods to download PDF files from the web,
    validate the content type, handle errors, and save files locally.
    It includes robust error handling and progress tracking.
    """
    @staticmethod
    def download_pdf_from_link(url: str, output_path: Optional[str] = None, filename: Optional[str] = None) -> Tuple[Optional[str], str]:
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
                            if file_size > 0:  # Only show progress if we know the file size
                                print(f"Downloaded: {downloaded}/{file_size} bytes ({downloaded/file_size*100:.1f}%)", end='\r')
                
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
            print(f"Unexpected error in download_pdf_from_link: {str(e)}")
            print(traceback.format_exc())
            return None, f"Unexpected error: {str(e)}"


class AnnualReportAgent:
    """
    Agent for finding and downloading company annual reports.
    
    This class uses AI capabilities to locate and retrieve company
    annual reports by finding investor relations pages and identifying
    the appropriate PDF links. It handles the complexities of navigating
    corporate websites and finding the right documents.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the annual report retrieval agent.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided,
                                    will try to get from environment variables.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI handler
        self.openai_handler = OpenAIHandler(self.api_key)
    
    def find_and_download_annual_report(self, company_name: str) -> Optional[str]:
        """
        Find and download the latest annual report for a company.
        
        Args:
            company_name (str): The name of the company
            
        Returns:
            str or None: Path to the downloaded PDF, or None if failed
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
            dummy_filepath = os.path.join(os.getcwd(), f"{company_name.lower().replace(' ', '_')}_annual_report_info.txt")
            with open(dummy_filepath, 'w') as f:
                f.write(f"Please visit {ir_url} to access {company_name}'s annual reports.")
            
            return dummy_filepath
            
        except Exception as e:
            print(f"Error in annual report agent: {e}")
            return None


class PDFRequestParser:
    """
    Parses and detects user intentions related to PDF operations.
    
    This class contains methods to analyze user messages for PDF-related
    requests, categorize the type of request, and extract relevant details
    such as URLs, companies, or other parameters.
    """
    @staticmethod
    def is_asking_for_pdf_download(text: str) -> bool:
        """
        Check if the user is asking for a direct PDF download.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking for PDF download, False otherwise.
        """
        pdf_keywords = [
            "pdf", "download pdf", "get pdf", "pdf file", "document",
            "report", "annual report", "download report", "download document",
            "financial report", "financial document", "whitepaper"
        ]
        
        url_keywords = [
            "url", "link", "website", "address", "online", "internet", "web page",
            "web address", "download from", "get from"
        ]
        
        has_pdf = any(kw in text.lower() for kw in pdf_keywords)
        has_url = any(kw in text.lower() for kw in url_keywords)
        
        return has_pdf and has_url
    
    @staticmethod
    def is_asking_for_marketing_data(text: str) -> bool:
        """
        Check if the user is asking for marketing data or PDFs.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking for marketing data, False otherwise.
        """
        marketing_keywords = [
            "marketing", "market data", "market analysis", "market research",
            "marketing report", "marketing pdf", "marketing document", 
            "advertising data", "promotion", "campaign", "marketing materials",
            "marketing strategy", "marketing plan", "sales materials"
        ]
        
        return any(kw in text.lower() for kw in marketing_keywords)
    
    @staticmethod
    def extract_company_name(text: str) -> Optional[str]:
        """
        Extract a company name from the user's input.
        
        Args:
            text (str): The user's message.
            
        Returns:
            str or None: The extracted company name, or None if no company name found.
        """
        patterns = [
            r"(?:with|against|to|and)\s+([A-Z][A-Za-z\s]+?)(?:'s)?\s+(?:performance|report|company)",
            r"(?:compare|comparison|comparing)\s+(?:with|to|against)?\s+([A-Z][A-Za-z\s]+)",
            r"([A-Z][A-Za-z\s]+?)(?:'s)?\s+(?:annual report|report|performance)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None