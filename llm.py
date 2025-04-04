import os
import json
import requests
from typing import Optional


class OpenAIHandler:
    """
    Handles interactions with the OpenAI API.
    
    This class provides methods to send prompts to OpenAI models,
    retrieve responses, and generate Python code for data analysis.
    It handles API key management, request formatting, and error handling.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI handler with the provided API key.
        
        Args:
            api_key (str, optional): The OpenAI API key. If not provided,
                                    will try to get from environment variables.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
    def set_api_key(self, api_key: str):
        """
        Set or update the OpenAI API key.
        
        Args:
            api_key (str): The new OpenAI API key to use.
        """
        self.api_key = api_key
        
    def get_response(self, prompt: str, system_message: str = "", model: str = "gpt-4"):
        """
        Get a response from OpenAI API.
        
        Args:
            prompt (str): The user prompt to send to the model.
            system_message (str, optional): The system message to set the context.
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4".
            
        Returns:
            str: The model's response or an error message.
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
    
    def generate_analysis_code(self, user_prompt: str, file_info: str, file_preview: Optional[str] = None):
        """
        Generate Python code for data analysis based on user prompt and file info.
        
        Args:
            user_prompt (str): The user's request for what analysis to perform.
            file_info (str): Information about the file being analyzed.
            file_preview (str, optional): Preview of the file content.
            
        Returns:
            str: Generated Python code that performs the requested analysis.
        """
        system_message = """
        You are an expert data analyst AI assistant. Your task is to generate Python code to analyze data based on the user's request.
        The code you generate will be executed in a Python environment with the following libraries available:
        - pandas (as pd)
        - numpy (as np)
        - matplotlib.pyplot (as plt)
        - seaborn (sns)

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


class MessageHandler:
    """
    Handles processing and analysis of user messages.
    
    This class contains methods to detect intent from user messages,
    extract relevant information, and determine the appropriate action
    to take based on the message content.
    """
    @staticmethod
    def extract_code(markdown_text: str) -> Optional[str]:
        """
        Extract code blocks from markdown text.
        
        Args:
            markdown_text (str): The markdown text containing code blocks.
            
        Returns:
            str or None: The extracted code without markdown formatting, or None if no code found.
        """
        import re
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        return None
    

    @staticmethod
    def is_asking_for_duplicate_detection(text: str) -> bool:
        """
        Check if the user is asking to detect duplicate records.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking for duplicate detection, False otherwise.
        """
        duplicate_keywords = [
            "find duplicate", "detect duplicate", "identify duplicate", 
            "duplicate record", "duplicate entry", "duplicate data",
            "similar record", "similar entry", "find similar", 
            "record deduplication", "deduplicate", "how many duplicate",
            "check for duplicate", "duplicate check"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in duplicate_keywords)

    @staticmethod
    def clean_response(response: str) -> str:
        """
        Remove code blocks from display text for cleaner conversation rendering.
        
        Args:
            response (str): The original response text with code blocks.
            
        Returns:
            str: The cleaned response with code blocks replaced by a placeholder.
        """
        import re
        clean_text = re.sub(r"```(?:python)?(.*?)```", "[Code generated and executed]", response, flags=re.DOTALL)
        return clean_text
    
    @staticmethod
    def is_asking_to_clear_chat(text: str) -> bool:
        """
        Check if the user is asking to clear the chat history.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking to clear chat, False otherwise.
        """
        clear_chat_keywords = [
            "clear chat", "clear my chat", "clear the chat", "clear conversation",
            "clear history", "clear chat history", "reset chat", "start new chat",
            "wipe chat", "erase chat", "delete chat", "delete conversation",
            "new conversation", "clean chat", "clean history"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in clear_chat_keywords)
    
    @staticmethod
    def is_asking_for_data_quality_check(text: str) -> bool:
        """
        Check if the user is asking for a data quality check.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking for data quality check, False otherwise.
        """
        quality_check_keywords = [
            "check data quality", "check quality", "data quality", "check data",
            "quality check", "quality report", "data health", "health check",
            "data validation", "validate data", "data profiling", "profile data",
            "data assessment", "assess data", "data audit", "audit data"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in quality_check_keywords)
    
    @staticmethod
    def is_asking_for_dataframe_operations(text: str) -> bool:
        """
        Check if the user is asking for DataFrame operations like merge, join, etc.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking for DataFrame operations, False otherwise.
        """
        df_operation_keywords = [
            "merge", "join", "concat", "combine", "melt", "pivot", 
            "fill na", "fillna", "drop na", "dropna", "replace na",
            "groupby", "group by", "aggregate", "agg", "sort", "filter",
            "transform", "convert", "reshape", "stack", "unstack"
        ]
        
        text_lower = text.lower()
        
        # Check if user is asking about dataframes or multiple files
        has_df_context = any(kw in text_lower for kw in ["dataframe", "data frame", "df", "datasets", "files"])
        
        # Check if any operation keywords are present
        has_operation = any(kw in text_lower for kw in df_operation_keywords)
        
        return has_operation and has_df_context
    
    @staticmethod
    def is_asking_for_comparison(text: str) -> bool:
        """
        Check if the user is asking to compare reports or companies.
        
        Args:
            text (str): The user's message.
            
        Returns:
            bool: True if the user is asking for a comparison, False otherwise.
        """
        comparison_keywords = [
            "compare", "comparison", "comparing", "benchmark", "benchmarking",
            "versus", "vs", "against", "performance comparison"
        ]
        report_keywords = [
            "report", "annual report", "financial report", "performance", "financials",
            "statements", "financial statements", "10-k", "10k"
        ]
        company_keywords = [
            "company", "organization", "organisation", "competitor", "competitors",
            "business", "corporation", "enterprise", "firm"
        ]
        
        has_comparison = any(kw in text.lower() for kw in comparison_keywords)
        has_report = any(kw in text.lower() for kw in report_keywords)
        has_company = any(kw in text.lower() for kw in company_keywords)
        
        return has_comparison and (has_report or has_company)
    
    @staticmethod
    def is_asking_for_marketing_data(text: str) -> bool:
        """
        Check if the user is asking for marketing data.
        
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
    def is_asking_about_pdf(text: str, pdf_processed: bool) -> bool:
        """
        Check if the user is asking a question about a processed PDF.
        
        Args:
            text (str): The user's message.
            pdf_processed (bool): Whether a PDF has been processed in the current session.
            
        Returns:
            bool: True if the user is asking about the PDF, False otherwise.
        """
        # If we haven't processed a PDF yet, this isn't a PDF question
        if not pdf_processed:
            return False
        
        # Check for question indicators
        question_indicators = ["?", "what", "who", "where", "when", "why", "how", "tell me", "show me", "explain", "describe"]
        has_question = any(indicator in text.lower() for indicator in question_indicators)
        
        # Check for PDF references
        pdf_references = [
            "pdf", "document", "report", "annual report", "file", "content",
            "in the", "from the", "according to", "says", "mention", "state"
        ]
        has_pdf_ref = any(ref in text.lower() for ref in pdf_references)
        
        # If it's clearly a PDF question
        if has_question and has_pdf_ref:
            return True
        
        # If it's just a question and we've processed a PDF, assume it's about the PDF
        # unless it's clearly about something else
        if has_question and not any(x in text.lower() for x in ["load file", "export", "download", "data analysis"]):
            return True
        
        return False
    
    @staticmethod
    def extract_company_name(text: str) -> Optional[str]:
        """
        Extract a company name from the user's input.
        
        Args:
            text (str): The user's message.
            
        Returns:
            str or None: The extracted company name, or None if no company name found.
        """
        import re
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