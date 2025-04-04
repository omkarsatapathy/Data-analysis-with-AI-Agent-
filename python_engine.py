import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CodeExecutor:
    """
    Handles Python code execution and captures results.
    
    This class provides methods to execute Python code generated by AI models
    in a controlled environment. It captures standard output, errors, and any
    generated matplotlib figures, making it suitable for interactive data analysis
    and visualization in a web application context.
    """
    @staticmethod
    def execute_code(code: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute Python code and capture output and figures.
        
        Args:
            code (str): Python code to execute
            df (DataFrame, optional): DataFrame to make available to the code
            
        Returns:
            dict: Execution results including outputs, errors, figures, and local variables
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


class DataFrameFilterExecutor:
    """
    Specialized executor for filtering DataFrames based on user requests.
    
    This class provides methods to generate and execute Python code
    specifically for filtering DataFrames based on user criteria.
    It works with an AI model to interpret natural language filtering
    requests and turn them into executable Python code.
    """
    @staticmethod
    def execute_filter_request(
        df: pd.DataFrame, 
        filter_request: str, 
        openai_handler: Any
    ) -> Dict[str, Any]:
        """
        Generate and execute code to filter a DataFrame based on user request.
        
        Args:
            df (DataFrame): DataFrame to filter
            filter_request (str): User request in natural language
            openai_handler: Object that handles OpenAI API requests
            
        Returns:
            dict: Results of the filtering operation
        """
        if not isinstance(df, pd.DataFrame):
            return {
                'output': '',
                'errors': 'No valid DataFrame provided for filtering',
                'filtered_df': None,
                'success': False
            }
        
        # Generate filtering code
        system_message = """
        You are an expert data analyst AI assistant. Generate Python code to filter a DataFrame based on the user's request.
        The DataFrame is already loaded as 'df'. The code should return a filtered DataFrame.
        Respond with only Python code inside triple backticks. The code should end with the filtered DataFrame assigned to a variable called 'filtered_df'.
        """
        
        # Create prompt with DataFrame info
        columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
        prompt = f"""
        USER REQUEST: {filter_request}
        
        DATAFRAME INFO:
        - Shape: {df.shape}
        - Columns: {columns_info}
        
        Generate Python code to filter the data as requested. The filtered DataFrame should be assigned to 'filtered_df'.
        """
        
        # Get code from AI
        response = openai_handler.get_response(prompt, system_message)
        
        # Extract code from the response using a simple regex
        import re
        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        
        if not code_match:
            return {
                'output': 'Could not extract code from AI response',
                'errors': 'No code block found in response',
                'code': response,
                'filtered_df': None,
                'success': False
            }
        
        code = code_match.group(1).strip()
        
        # Execute the filtering code
        result = CodeExecutor.execute_code(code, df)
        
        # Check if filtered_df was created
        if result['success'] and 'filtered_df' in result.get('local_vars', {}):
            filtered_df = result['local_vars']['filtered_df']
            
            # Basic validation of filtered_df
            if isinstance(filtered_df, pd.DataFrame):
                return {
                    'output': result['output'],
                    'errors': result['errors'],
                    'code': code,
                    'filtered_df': filtered_df,
                    'success': True
                }
        
        # If we got here, filtering failed
        return {
            'output': result.get('output', ''),
            'errors': result.get('errors', 'Failed to create filtered_df'),
            'code': code,
            'filtered_df': None,
            'success': False
        }


class DataVisualizationExecutor:
    """
    Specialized executor for creating data visualizations based on user requests.
    
    This class provides methods to generate and execute Python code
    specifically for creating data visualizations using matplotlib and seaborn.
    It works with an AI model to interpret natural language visualization
    requests and turn them into executable Python code.
    """
    @staticmethod
    def create_visualization(
        df: pd.DataFrame, 
        viz_request: str, 
        openai_handler: Any
    ) -> Dict[str, Any]:
        """
        Generate and execute code to create visualizations based on user request.
        
        Args:
            df (DataFrame): DataFrame to visualize
            viz_request (str): User request in natural language
            openai_handler: Object that handles OpenAI API requests
            
        Returns:
            dict: Results including any generated figures
        """
        if not isinstance(df, pd.DataFrame):
            return {
                'output': '',
                'errors': 'No valid DataFrame provided for visualization',
                'figures': [],
                'success': False
            }
        
        # Generate visualization code
        system_message = """
        You are an expert data visualization AI assistant. Generate Python code to create clear, informative visualizations based on the user's request.
        The DataFrame is already loaded as 'df'. Use matplotlib and seaborn to create the visualizations.
        Follow these guidelines:
        1. Always use plt.figure() to create a new figure with appropriate size
        2. Set clear titles, labels, and legends
        3. Use appropriate color schemes for the data type
        4. Always include plt.tight_layout() and plt.show()
        5. Write clean, well-commented code
        
        Respond with only Python code inside triple backticks.
        """
        
        # Create prompt with DataFrame info
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
        
        prompt = f"""
        USER REQUEST: {viz_request}
        
        DATAFRAME INFO:
        - Shape: {df.shape}
        - Numeric columns: {numeric_cols}
        - Categorical columns: {categorical_cols}
        - Datetime columns: {datetime_cols}
        
        Generate Python code to create appropriate visualizations based on this request.
        """
        
        # Get code from AI
        response = openai_handler.get_response(prompt, system_message)
        
        # Extract code from the response
        import re
        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        
        if not code_match:
            return {
                'output': 'Could not extract visualization code from AI response',
                'errors': 'No code block found in response',
                'code': response,
                'figures': [],
                'success': False
            }
        
        code = code_match.group(1).strip()
        
        # Execute the visualization code
        result = CodeExecutor.execute_code(code, df)
        
        return {
            'output': result.get('output', ''),
            'errors': result.get('errors', ''),
            'code': code,
            'figures': result.get('figures', []),
            'success': result.get('success', False)
        }