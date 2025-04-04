import os
import pandas as pd
from typing import Tuple, Dict, Any, Optional


class FileExporter:
    """
    Handles exporting DataFrames to various file formats.
    
    This class provides methods to export pandas DataFrames to different
    file formats such as CSV, Excel, and Parquet. It handles file path
    creation, directory validation, and provides appropriate feedback
    and error handling.
    """
    @staticmethod
    def export_dataframe(
        df: pd.DataFrame, 
        file_path: str, 
        file_format: str = 'csv'
    ) -> Tuple[str, bool]:
        """
        Export DataFrame to the specified file format.
        
        Args:
            df (DataFrame): The pandas DataFrame to export
            file_path (str): Path where the file should be saved
            file_format (str): Format to export (csv, excel, xlsx, parquet)
            
        Returns:
            tuple: (message, success_flag)
        """
        try:
            # Check if DataFrame is valid
            if df is None or not isinstance(df, pd.DataFrame):
                return "Error: Invalid DataFrame provided for export.", False
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Normalize file format
            file_format = file_format.lower()
            
            # Handle CSV export
            if file_format == 'csv':
                return FileExporter._export_csv(df, file_path)
                
            # Handle Excel export
            elif file_format in ['excel', 'xlsx', 'xls']:
                return FileExporter._export_excel(df, file_path)
                
            # Handle Parquet export
            elif file_format == 'parquet':
                return FileExporter._export_parquet(df, file_path)
            
            else:
                return f"Unsupported export format: {file_format}", False
                
        except Exception as e:
            return f"Error exporting file: {str(e)}", False
    
    @staticmethod
    def _export_csv(df: pd.DataFrame, file_path: str) -> Tuple[str, bool]:
        """
        Export DataFrame to CSV format.
        
        Args:
            df (DataFrame): The pandas DataFrame to export
            file_path (str): Path where the file should be saved
            
        Returns:
            tuple: (message, success_flag)
        """
        # Ensure the file has the correct extension
        if not file_path.lower().endswith('.csv'):
            file_path = f"{file_path}.csv"
        
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        # Verify the file was created
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return f"Successfully exported DataFrame to {file_path}", True
        else:
            return f"Export process completed but the file {file_path} appears to be empty or not created.", False
    
    @staticmethod
    def _export_excel(df: pd.DataFrame, file_path: str) -> Tuple[str, bool]:
        """
        Export DataFrame to Excel format.
        
        Args:
            df (DataFrame): The pandas DataFrame to export
            file_path (str): Path where the file should be saved
            
        Returns:
            tuple: (message, success_flag)
        """
        # For Excel files, ensure it has .xlsx extension
        if not any(file_path.lower().endswith(ext) for ext in ['.xlsx', '.xls']):
            file_path = f"{file_path}.xlsx"
        
        try:
            # Try to use openpyxl engine for .xlsx files
            df.to_excel(file_path, index=False, engine='openpyxl')
            
            # Verify the file was created
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return f"Successfully exported DataFrame to {file_path}", True
            else:
                return f"Export process completed but the file {file_path} appears to be empty or not created.", False
                
        except ImportError:
            return "Error: openpyxl is required for Excel export. Install it with 'pip install openpyxl'", False
    
    @staticmethod
    def _export_parquet(df: pd.DataFrame, file_path: str) -> Tuple[str, bool]:
        """
        Export DataFrame to Parquet format.
        
        Args:
            df (DataFrame): The pandas DataFrame to export
            file_path (str): Path where the file should be saved
            
        Returns:
            tuple: (message, success_flag)
        """
        # Ensure the file has the correct extension
        if not file_path.lower().endswith('.parquet'):
            file_path = f"{file_path}.parquet"
        
        try:
            df.to_parquet(file_path, index=False)
            
            # Verify the file was created
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return f"Successfully exported DataFrame to {file_path}", True
            else:
                return f"Export process completed but the file {file_path} appears to be empty or not created.", False
                
        except ImportError:
            return "Error: pyarrow or fastparquet is required for Parquet export. Install with 'pip install pyarrow'", False


class FileHandler:
    """
    Handles various file operations like loading, parsing, and exporting.
    
    This class provides methods to load files of different formats into 
    pandas DataFrames, extract metadata and statistics from files, and
    determine file types based on extensions.
    """
    @staticmethod
    def load_file(file_path: str) -> Tuple[Any, str]:
        """
        Load file from given path and return content and type.
        
        Args:
            file_path (str): Path to the file to load
            
        Returns:
            tuple: (content, file_type) or (error_message, 'error')
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
                    import json
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
    def get_file_info(df: pd.DataFrame) -> Dict[str, str]:
        """
        Get information about a DataFrame.
        
        Args:
            df (DataFrame): The DataFrame to analyze
            
        Returns:
            dict: Dictionary containing DataFrame metadata and statistics
        """
        if not isinstance(df, pd.DataFrame):
            return {"error": "Not a DataFrame"}
        
        import io
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
    def filter_dataframe(df: pd.DataFrame, filter_code: Optional[str] = None) -> pd.DataFrame:
        """
        Filter a DataFrame based on provided Python code.
        
        Args:
            df (DataFrame): DataFrame to filter
            filter_code (str, optional): Python code for filtering
            
        Returns:
            DataFrame: Filtered DataFrame or original if no code provided
        """
        if filter_code is None or not isinstance(df, pd.DataFrame):
            return df
        
        # Create a namespace for execution
        locals_dict = {'df': df.copy(), 'pd': pd}
        
        try:
            # Execute the filtering code
            exec(filter_code, globals(), locals_dict)
            
            # Check if filtered_df was created
            if 'filtered_df' in locals_dict and isinstance(locals_dict['filtered_df'], pd.DataFrame):
                return locals_dict['filtered_df']
            else:
                print("Warning: filter_code did not produce 'filtered_df'. Returning original.")
                return df
        except Exception as e:
            print(f"Error in filtering code: {str(e)}")
            return df