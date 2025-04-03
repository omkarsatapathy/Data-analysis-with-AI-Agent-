import streamlit as st
import os
import pandas as pd
import numpy as np
from Utils import OpenAIHandler, FileHandler, CodeExecutor, AnnualReportAgent, PDFParser, RAGSystem
import re
import uuid

# Set page configuration
st.set_page_config(
    page_title="Agentic AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
# Add support for multiple files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}  # dict to store multiple files {id: {name, content, type}}
if "active_file_id" not in st.session_state:
    st.session_state.active_file_id = None  # currently selected file
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if "show_file_input" not in st.session_state:
    st.session_state.show_file_input = False
if "show_export_input" not in st.session_state:
    st.session_state.show_export_input = False
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "code_execution_result" not in st.session_state:
    st.session_state.code_execution_result = None
if "show_company_input" not in st.session_state:
    st.session_state.show_company_input = False
if "annual_report_path" not in st.session_state:
    st.session_state.annual_report_path = None
if "comparison_company" not in st.session_state:
    st.session_state.comparison_company = None
# Add new session state variables for PDF URL download
if "show_pdf_url_input" not in st.session_state:
    st.session_state.show_pdf_url_input = False
if "pdf_download_purpose" not in st.session_state:
    st.session_state.pdf_download_purpose = None
# Add RAG system session state variables
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None
# Add DataFrame operation variables
if "show_df_operations" not in st.session_state:
    st.session_state.show_df_operations = False
if "operation_result" not in st.session_state:
    st.session_state.operation_result = None
if "operation_description" not in st.session_state:
    st.session_state.operation_description = None

# Initialize the OpenAI handler
openai_handler = OpenAIHandler(st.session_state.openai_api_key)

# Function to extract code from markdown
def extract_code(markdown_text):
    # Pattern to match code blocks with or without language specification
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return None

# Function to clean up conversation display
def clean_response(response):
    # Remove code blocks from display
    clean_text = re.sub(r"```(?:python)?(.*?)```", "[Code generated and executed]", response, flags=re.DOTALL)
    return clean_text

# Function to check if the user is asking to clear chat
def is_asking_to_clear_chat(text):
    """
    Check if the user is asking to clear the chat history
    """
    clear_chat_keywords = [
        "clear chat", "clear my chat", "clear the chat", "clear conversation",
        "clear history", "clear chat history", "reset chat", "start new chat",
        "wipe chat", "erase chat", "delete chat", "delete conversation",
        "new conversation", "clean chat", "clean history"
    ]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check if any of the clear chat keywords are in the user's message
    return any(keyword in text_lower for keyword in clear_chat_keywords)

# Function to check if the user is asking for data quality check
def is_asking_for_data_quality_check(text):
    """
    Check if the user is asking for a data quality check
    """
    quality_check_keywords = [
        "check data quality", "check quality", "data quality", "check data",
        "quality check", "quality report", "data health", "health check",
        "data validation", "validate data", "data profiling", "profile data",
        "data assessment", "assess data", "data audit", "audit data"
    ]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check if any of the quality check keywords are in the user's message
    return any(keyword in text_lower for keyword in quality_check_keywords)

# Function to check if user is asking for DataFrame operations
def is_asking_for_dataframe_operations(text):
    """
    Check if the user is asking for DataFrame operations like merge, join, etc.
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
    
    return has_operation and (has_df_context or len(st.session_state.uploaded_files) > 1)

# Function to generate a data quality report
def generate_data_quality_report(df):
    """
    Generate a comprehensive data quality report for the DataFrame
    
    Args:
        df (pandas.DataFrame): The DataFrame to analyze
        
    Returns:
        str: A formatted data quality report
    """
    if not isinstance(df, pd.DataFrame):
        return "Error: No DataFrame to analyze"
    
    # Basic information
    row_count = df.shape[0]
    column_count = df.shape[1]
    
    # NaN information
    nan_counts = df.isna().sum()
    non_nan_counts = df.notna().sum()
    
    # Calculate percentages
    if row_count > 0:
        nan_percentages = (nan_counts / row_count * 100).round(2)
    else:
        nan_percentages = nan_counts * 0
    
    # Find column with highest percentage of NaNs
    if not nan_percentages.empty and nan_percentages.max() > 0:
        max_nan_col = nan_percentages.idxmax()
        max_nan_percentage = nan_percentages.max()
    else:
        max_nan_col = "None"
        max_nan_percentage = 0
    
    # Data type information
    dtypes = df.dtypes
    
    # Special check for phone numbers
    phone_number_cols = []
    phone_quality_info = []
    
    # Look for columns that might contain phone numbers
    for col in df.columns:
        col_lower = str(col).lower()
        if any(phone_term in col_lower for phone_term in ["phone", "mobile", "cell", "tel", "contact"]):
            phone_number_cols.append(col)
    
    # Analyze phone number columns if any
    for col in phone_number_cols:
        # Skip if column has all NaN values
        if nan_counts[col] == row_count:
            continue
        
        # Only analyze string columns or columns that could contain phone numbers
        if df[col].dtype == 'object' or 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
            # Convert to string and clean
            phone_data = df[col].fillna('').astype(str)
            
            # Count digits in each phone number
            digit_counts = phone_data.apply(lambda x: sum(c.isdigit() for c in x))
            
            # Basic phone validation (only count non-empty values)
            non_empty = (phone_data != '').sum()
            valid_10_digit = ((digit_counts == 10) & (phone_data.str.len() >= 10)).sum()
            valid_with_country_code = ((digit_counts > 10) & (digit_counts <= 15)).sum()
            invalid_phones = non_empty - valid_10_digit - valid_with_country_code
            
            if non_empty > 0:
                phone_info = f"""
            Column '{col}' phone number analysis:
            - Valid 10-digit numbers: {valid_10_digit} ({valid_10_digit/non_empty*100:.2f}% of non-empty values)
            - Valid with country code: {valid_with_country_code} ({valid_with_country_code/non_empty*100:.2f}% of non-empty values)
            - Invalid or unusual formats: {invalid_phones} ({invalid_phones/non_empty*100:.2f}% of non-empty values)
                """
                phone_quality_info.append(phone_info)
    
    # Build the report
    report = f"""
    # DATA QUALITY REPORT
    
    ## Basic Information
    - Total rows: {row_count:,}
    - Total columns: {column_count}
    
    ## Missing Data Summary
    - Total missing values: {nan_counts.sum():,}
    - Column with most missing values: '{max_nan_col}' ({max_nan_percentage:.2f}% missing)
    
    ## Column-by-Column Analysis
    """
    
    # Add column details
    for col in df.columns:
        non_null_percent = non_nan_counts[col]/row_count*100 if row_count > 0 else 0
        report += f"""
    '{col}':
      - Data type: {dtypes[col]}
      - Non-null values: {non_nan_counts[col]:,} ({non_null_percent:.2f}%)
      - Missing values: {nan_counts[col]:,} ({nan_percentages[col]:.2f}%)
        """
        
        # For numeric columns, add basic stats
        if pd.api.types.is_numeric_dtype(df[col]):
            if non_nan_counts[col] > 0:
                try:
                    report += f"""
      - Min: {df[col].min()}
      - Max: {df[col].max()}
      - Mean: {df[col].mean():.2f}
      - Unique values: {df[col].nunique():,}
                    """
                except:
                    # In case of any calculation errors
                    report += f"""
      - Unable to calculate statistics for this column
                    """
        # For categorical/text columns
        elif df[col].dtype == 'object':
            if non_nan_counts[col] > 0:
                try:
                    value_counts = df[col].value_counts()
                    report += f"""
      - Unique values: {df[col].nunique():,} ({df[col].nunique()/row_count*100:.2f}% of total rows)
                    """
                    if not value_counts.empty:
                        report += f"""
      - Most common value: '{value_counts.index[0]}' (occurs {value_counts.iloc[0]:,} times)
                        """
                except:
                    report += f"""
      - Unable to analyze categorical values
                    """
    
    # Add phone number quality info if any
    if phone_quality_info:
        report += "\n    ## Phone Number Quality Analysis\n"
        for info in phone_quality_info:
            report += info
    
    # Add data quality score
    # Simple score based on completeness (100% - average percentage of missing values)
    avg_missing_percentage = nan_percentages.mean() if not nan_percentages.empty else 0
    quality_score = 100 - avg_missing_percentage
    
    report += f"""
    ## Overall Data Quality Score
    - Completeness score: {quality_score:.2f}/100
    """
    
    # Add duplicate check
    try:
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / row_count * 100) if row_count > 0 else 0
        report += f"""
    ## Duplicate Analysis
    - Duplicate rows: {duplicate_rows:,} ({duplicate_percentage:.2f}% of total rows)
        """
    except:
        report += """
    ## Duplicate Analysis
    - Unable to check for duplicates
        """
    
    return report

# Function to check if the user is asking to compare reports
def is_asking_for_comparison(text):
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

# Function to check if the user is asking for marketing data
def is_asking_for_marketing_data(text):
    marketing_keywords = [
        "marketing", "market data", "market analysis", "market research",
        "marketing report", "marketing pdf", "marketing document", 
        "advertising data", "promotion", "campaign", "marketing materials",
        "marketing strategy", "marketing plan", "sales materials"
    ]
    
    return any(kw in text.lower() for kw in marketing_keywords)

# Function to check if the user is asking for a direct PDF download
def is_asking_for_pdf_download(text):
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

# Function to check if the user is asking a question about the PDF
def is_asking_about_pdf(text):
    # If we haven't processed a PDF yet, this isn't a PDF question
    if not st.session_state.pdf_processed:
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

# Function to get a descriptive summary of a DataFrame
def get_dataframe_summary(df, name="DataFrame"):
    """
    Generate a comprehensive summary of a DataFrame
    
    Args:
        df (pandas.DataFrame): The DataFrame to summarize
        name (str): Name to use for the DataFrame in the summary
        
    Returns:
        str: A formatted summary of the DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        return f"Error: {name} is not a valid DataFrame"
    
    # Basic information
    summary = f"""
    # {name} Summary
    
    ## Basic Information
    - Shape: {df.shape[0]} rows × {df.shape[1]} columns
    - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    
    ## Column Information
    """
    
    # Column information
    for col in df.columns:
        dtype = df[col].dtype
        na_count = df[col].isna().sum()
        na_percent = (na_count / len(df) * 100) if len(df) > 0 else 0
        
        summary += f"""
    - {col} ({dtype})
      - Missing values: {na_count} ({na_percent:.2f}%)
    """
        
        # Add statistics based on column type
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].count() > 0:  # Only add stats if there are non-NA values
                summary += f"""
      - Min: {df[col].min()}
      - Max: {df[col].max()}
      - Mean: {df[col].mean():.4f}
      - Std Dev: {df[col].std():.4f}
    """
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            unique_count = df[col].nunique()
            summary += f"""
      - Unique values: {unique_count}
    """
            if unique_count < 10 and unique_count > 0:  # Show value counts for low cardinality columns
                value_counts = df[col].value_counts().head(5)
                summary += "      - Top values:\n"
                for val, count in value_counts.items():
                    pct = (count / df[col].count() * 100) if df[col].count() > 0 else 0
                    summary += f"        - {val}: {count} ({pct:.2f}%)\n"
    
    # Add head of DataFrame
    summary += f"""
    ## Data Preview (First 5 rows)
    {df.head().to_string()}
    """
    
    return summary

# Function to perform DataFrame operations based on user input
def perform_dataframe_operation(operation_type, params, file_ids):
    """
    Perform DataFrame operations like merge, join, concatenate, etc.
    
    Args:
        operation_type (str): Type of operation (merge, join, concat, etc.)
        params (dict): Parameters for the operation
        file_ids (list): IDs of files to operate on
        
    Returns:
        tuple: (resulting DataFrame, operation description)
    """
    # Get the DataFrames to operate on
    dfs = []
    df_names = []
    
    for file_id in file_ids:
        if file_id in st.session_state.uploaded_files:
            file_info = st.session_state.uploaded_files[file_id]
            dfs.append(file_info['content'])
            df_names.append(file_info['name'])
    
    if len(dfs) == 0:
        return None, "No DataFrames selected for operation"
    
    try:
        result_df = None
        description = ""
        
        if operation_type == "merge":
            left_df = dfs[0]
            right_df = dfs[1] if len(dfs) > 1 else dfs[0]
            
            # Get merge parameters
            left_on = params.get('left_on')
            right_on = params.get('right_on', left_on)
            how = params.get('how', 'inner')
            simple_mode = params.get('simple_mode', False)
            
            # For simple mode (column fetch)
            if simple_mode:
                # Get fetch columns
                fetch_columns = params.get('fetch_columns', [])
                handle_dupes = params.get('handle_dupes', 'Keep first occurrence only')
                
                # Check for and handle duplicates if needed
                if handle_dupes in ['Keep first occurrence only', 'Keep last occurrence only']:
                    # Deduplicate the right dataframe on the join key
                    keep_option = 'first' if handle_dupes == 'Keep first occurrence only' else 'last'
                    right_df = right_df.drop_duplicates(subset=[right_on], keep=keep_option)
                    
                    # Also deduplicate left dataframe if inner join to prevent duplicates
                    if how == 'inner':
                        left_df = left_df.drop_duplicates(subset=[left_on], keep=keep_option)
                
                # Create a right dataframe with only the columns we need
                if fetch_columns:
                    columns_to_fetch = [right_on] + fetch_columns
                    right_df_subset = right_df[columns_to_fetch]
                else:
                    right_df_subset = right_df
                
                # Perform the join
                result_df = pd.merge(
                    left_df,
                    right_df_subset,
                    left_on=left_on,
                    right_on=right_on,
                    how=how
                )
                
                # Remove the duplicate ID column from the right dataframe
                if right_on in result_df.columns and f"{right_on}_x" in result_df.columns:
                    # Rename columns to avoid _x and _y suffixes
                    result_df = result_df.rename(columns={f"{right_on}_x": right_on})
                    # Drop the duplicate column
                    if f"{right_on}_y" in result_df.columns:
                        result_df = result_df.drop(columns=[f"{right_on}_y"])
                
                # Generate a user-friendly description
                fetch_desc = f"Fetched {len(fetch_columns)} columns" if fetch_columns else "Fetched all columns"
                how_desc = "Kept all rows from first file" if how == "left" else "Kept only matching rows"
                dedup_desc = ""
                if handle_dupes != "Keep all (may duplicate rows)":
                    dedup_desc = f"Removed duplicate IDs (kept {handle_dupes.lower()})"
                
                description = f"""
                # Column Fetch Operation
                
                ## What happened
                - Added columns from **{df_names[1]}** to **{df_names[0]}**
                - Matched on: **{left_on}** = **{right_on}**
                - {how_desc}
                - {fetch_desc}
                {f"- {dedup_desc}" if dedup_desc else ""}
                
                ## Columns fetched
                {', '.join(['**' + col + '**' for col in fetch_columns])}
                
                ## Result details
                - Original files: {df_names[0]} ({dfs[0].shape[0]} rows) and {df_names[1]} ({dfs[1].shape[0]} rows)
                - Result: {result_df.shape[0]} rows × {result_df.shape[1]} columns
                
                ## Resulting columns
                {list(result_df.columns)}
                """
            
            # For full join mode (original behavior)
            else:
                # New way - explicit fetch columns
                fetch_columns = params.get('fetch_columns', [])
                
                # Old way - right_cols
                right_cols = params.get('right_cols', [])
                
                # Prepare columns for the right dataframe
                columns_to_include = []
                
                # If fetch_columns is specified (new method), use those
                if fetch_columns:
                    # Always include the join column
                    columns_to_include = [right_on] + fetch_columns
                    # Filter the right dataframe 
                    filtered_right_df = right_df[columns_to_include]
                    
                    # Perform merge
                    result_df = pd.merge(
                        left_df, 
                        filtered_right_df,
                        left_on=left_on,
                        right_on=right_on,
                        how=how
                    )
                    
                    # Generate description in a more user-friendly way
                    description = f"""
                    # Merge Operation
                    
                    ## What happened
                    - Joined **{df_names[0]}** with **{df_names[1]}**
                    - Matched on: **{left_on}** = **{right_on}**
                    - Join type: **{how}**
                    
                    ## Columns fetched from {df_names[1]}
                    {', '.join(['**' + col + '**' for col in fetch_columns])}
                    
                    ## Result details
                    - Original shape: {df_names[0]} {dfs[0].shape}, {df_names[1]} {right_df.shape}
                    - Result shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns
                    
                    ## Resulting columns
                    {list(result_df.columns)}
                    """
                
                # Otherwise use the older right_cols method
                elif right_cols:
                    # Always include the join column if it's not in the selected columns
                    if right_on not in right_cols:
                        right_cols = [right_on] + right_cols
                    
                    # Filter the right dataframe to only include selected columns
                    filtered_right_df = right_df[right_cols]
                    
                    # Perform merge
                    result_df = pd.merge(
                        left_df, 
                        filtered_right_df,
                        left_on=left_on,
                        right_on=right_on,
                        how=how
                    )
                    
                    # Generate description
                    included_cols_info = "All columns" if not right_cols else f"Selected columns: {right_cols}"
                    
                    description = f"""
                    # Merge Operation
                    
                    - Left DataFrame: {df_names[0]} (shape: {dfs[0].shape})
                    - Right DataFrame: {df_names[1]} (shape: {right_df.shape})
                    - Merge type: {how}
                    - Left join key: {left_on}
                    - Right join key: {right_on}
                    - Right DataFrame columns included: {included_cols_info}
                    - Result shape: {result_df.shape}
                    
                    ## Common columns before merge:
                    {set(left_df.columns).intersection(set(right_df.columns))}
                    
                    ## Resulting columns:
                    {list(result_df.columns)}
                    """
                
                # If neither is specified, do a regular merge with all columns
                else:
                    # Perform merge with all columns
                    result_df = pd.merge(
                        left_df, 
                        right_df,
                        left_on=left_on,
                        right_on=right_on,
                        how=how
                    )
                    
                    # Generate description
                    description = f"""
                    # Merge Operation
                    
                    - Left DataFrame: {df_names[0]} (shape: {dfs[0].shape})
                    - Right DataFrame: {df_names[1]} (shape: {right_df.shape})
                    - Merge type: {how}
                    - Left join key: {left_on}
                    - Right join key: {right_on}
                    - Right DataFrame columns included: All columns
                    - Result shape: {result_df.shape}
                    
                    ## Common columns before merge:
                    {set(left_df.columns).intersection(set(right_df.columns))}
                    
                    ## Resulting columns:
                    {list(result_df.columns)}
                    """
            
        elif operation_type == "concat":
            axis = params.get('axis', 0)
            ignore_index = params.get('ignore_index', True)
            
            # Perform concatenation
            result_df = pd.concat(dfs, axis=axis, ignore_index=ignore_index)
            
            description = f"""
            # Concatenation Operation
            
            - DataFrames: {', '.join(df_names)}
            - Axis: {'rows (0)' if axis == 0 else 'columns (1)'}
            - Ignore index: {ignore_index}
            - Original shapes: {', '.join([str(df.shape) for df in dfs])}
            - Result shape: {result_df.shape}
            """
            
        elif operation_type == "fillna":
            df = dfs[0]
            value = params.get('value')
            method = params.get('method')
            
            if method:
                # Fill using method like 'ffill' or 'bfill'
                result_df = df.fillna(method=method)
                fill_desc = f"method='{method}'"
            else:
                # Fill using value
                result_df = df.fillna(value)
                fill_desc = f"value={value}"
            
            # Count NAs before and after
            na_before = df.isna().sum().sum()
            na_after = result_df.isna().sum().sum()
            
            description = f"""
            # Fill NA Operation
            
            - DataFrame: {df_names[0]} (shape: {df.shape})
            - Fill parameters: {fill_desc}
            - NAs before: {na_before}
            - NAs after: {na_after}
            - NAs filled: {na_before - na_after}
            """
            
        elif operation_type == "dropna":
            df = dfs[0]
            axis = params.get('axis', 0)
            how = params.get('how', 'any')
            
            # Perform drop NA
            result_df = df.dropna(axis=axis, how=how)
            
            description = f"""
            # Drop NA Operation
            
            - DataFrame: {df_names[0]}
            - Axis: {'rows (0)' if axis == 0 else 'columns (1)'}
            - How: {how}
            - Shape before: {df.shape}
            - Shape after: {result_df.shape}
            - {'Rows' if axis == 0 else 'Columns'} removed: {df.shape[axis] - result_df.shape[axis]}
            """
            
        elif operation_type == "melt":
            df = dfs[0]
            id_vars = params.get('id_vars', [])
            value_vars = params.get('value_vars', [col for col in df.columns if col not in id_vars])
            
            # Perform melt
            result_df = pd.melt(
                df,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=params.get('var_name', 'variable'),
                value_name=params.get('value_name', 'value')
            )
            
            description = f"""
            # Melt Operation
            
            - DataFrame: {df_names[0]}
            - ID variables: {id_vars}
            - Value variables: {len(value_vars)} columns
            - Shape before: {df.shape}
            - Shape after: {result_df.shape}
            """
            
        elif operation_type == "pivot":
            df = dfs[0]
            index = params.get('index')
            columns = params.get('columns')
            values = params.get('values')
            
            # Perform pivot
            if values:
                result_df = df.pivot(index=index, columns=columns, values=values)
            else:
                result_df = df.pivot(index=index, columns=columns)
            
            description = f"""
            # Pivot Operation
            
            - DataFrame: {df_names[0]}
            - Index: {index}
            - Columns: {columns}
            - Values: {values if values else 'All other columns'}
            - Shape before: {df.shape}
            - Shape after: {result_df.shape}
            """
            
        elif operation_type == "groupby":
            df = dfs[0]
            by = params.get('by', [])
            agg = params.get('agg', {})
            
            # Perform groupby
            result_df = df.groupby(by).agg(agg).reset_index()
            
            description = f"""
            # Group By Operation
            
            - DataFrame: {df_names[0]}
            - Grouping columns: {by}
            - Aggregations: {agg}
            - Shape before: {df.shape}
            - Shape after: {result_df.shape}
            - Number of groups: {len(result_df)}
            """
            
        else:
            return None, f"Unsupported operation: {operation_type}"
        
        # Add DataFrame summary
        full_description = description + "\n" + get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error performing {operation_type} operation: {str(e)}\n\nDetails:\n{error_details}"

# Function to process user message
def process_message(current_input):
    """Process the user message and generate a response"""
    if not current_input.strip():
        return False
    
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar")
        return False
        
    # Check if user is asking to clear chat
    if is_asking_to_clear_chat(current_input):
        # Keep the current user message
        user_message = {"role": "user", "content": current_input}
        
        # Clear the conversation history but save the user's clear request
        st.session_state.conversation = [user_message]
        
        # Add AI response confirming the chat has been cleared
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "Chat history has been cleared!"
        })
        
        return True
        
    # Check if user is asking for data quality check
    elif is_asking_for_data_quality_check(current_input):
        # Check if we have any files loaded
        if st.session_state.file_content is not None:
            # Generate data quality report for active file
            quality_report = generate_data_quality_report(st.session_state.file_content)
            
            # Add AI response with the quality report
            st.session_state.conversation.append({
                "role": "assistant",
                "content": quality_report
            })
            
            return True
        elif st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
            # Generate data quality report for the active file
            file_info = st.session_state.uploaded_files[st.session_state.active_file_id]
            quality_report = generate_data_quality_report(file_info['content'])
            
            # Add AI response with the quality report
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"# Data Quality Report for {file_info['name']}\n\n{quality_report}"
            })
            
            return True
        else:
            # No file loaded
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "No file is currently loaded. Please load a file first before requesting a data quality check."
            })
            
            return True
    
    # Check if user is asking for DataFrame operations
    elif is_asking_for_dataframe_operations(current_input):
        # Show DataFrame operations UI
        st.session_state.show_df_operations = True
        
        # Count available files
        available_files = len(st.session_state.uploaded_files)
        
        if available_files > 0:
            # Add AI response
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"I can help you perform DataFrame operations like merge, join, concat, fillna, and more. Please select the operation and files from the DataFrame Operations panel below."
            })
        else:
            # No files available
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "No files are currently loaded. Please upload files first before performing DataFrame operations."
            })
        
        return True
    
    # Handle PDF URL requests or preferences
    elif "provide a direct pdf url" in current_input.lower() or "provide pdf url" in current_input.lower():
        st.session_state.show_pdf_url_input = True
        st.session_state.pdf_download_purpose = "general"
        
        # Add AI response
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "Please enter the URL of the PDF file you'd like to download."
        })
        
        return True
    
    # Check if user is asking for direct PDF download
    elif is_asking_for_pdf_download(current_input):
        st.session_state.show_pdf_url_input = True
        st.session_state.pdf_download_purpose = "general"
        
        # Add AI response
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "I'd be happy to help you download a PDF. Please enter the URL of the PDF file you'd like to download."
        })
        
        return True
    
    # Check if user is asking for marketing data
    elif is_asking_for_marketing_data(current_input):
        st.session_state.show_pdf_url_input = True
        st.session_state.pdf_download_purpose = "marketing"
        
        # Add AI response
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "I'd be happy to help with marketing data. Please provide the URL of the marketing report or document you'd like me to download for analysis."
        })
        
        return True
    
    # Check if user is asking for company comparison
    elif is_asking_for_comparison(current_input):
        company_name = extract_company_name(current_input)
        
        if company_name:
            # Offer both options - auto-find or direct URL input
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"I'll help you compare with {company_name}'s annual report. Would you like me to automatically find the report or would you prefer to provide a direct PDF URL?"
            })
            return True
        else:
            # If no company name could be extracted, offer direct URL option
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "I'd be happy to help with the comparison. Would you like to provide a company name for me to find the report, or do you already have a PDF URL?"
            })
            return True
    
    # Check if user is asking to load a file
    elif "load" in current_input.lower() and any(word in current_input.lower() for word in ["csv", "excel", "xls", "xlsx", "parquet", "file", "data"]):
        # Show file input
        st.session_state.show_file_input = True
        
        # Add AI response
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "Please upload the file(s) you want to analyze. You can select multiple files by holding down the Ctrl (or Cmd) key while selecting."
        })
        
        return True
    
    # Check if user is asking to save/export a file
    elif any(word in current_input.lower() for word in ["save", "export", "write", "output"]) and any(word in current_input.lower() for word in ["csv", "excel", "xls", "xlsx", "parquet", "file", "data"]):
        df_to_export = None
        
        # Determine which DataFrame to export
        if st.session_state.operation_result is not None:
            # Use the result of the most recent DataFrame operation
            df_to_export = st.session_state.operation_result
        elif st.session_state.file_content is not None:
            # Use the traditional single file content
            df_to_export = st.session_state.file_content
        elif st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
            # Use the active file from multiple files
            df_to_export = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
        
        if df_to_export is not None:
            # Check if this is a filtering operation
            if any(word in current_input.lower() for word in ["filter", "where", "condition"]):
                # Generate filtering code
                file_info_dict = FileHandler.get_file_info(df_to_export)
                file_info = f"""
                DataFrame shape: {df_to_export.shape}
                Columns: {list(df_to_export.columns)}
                """
                
                # Special prompt for filtering data
                system_message = """
                You are an expert data analyst AI assistant. Generate Python code to filter a DataFrame based on the user's request.
                The DataFrame is already loaded as 'df'. The code should return a filtered DataFrame.
                Respond with only Python code inside triple backticks. The code should end with the filtered DataFrame assigned to a variable called 'filtered_df'.
                """
                
                prompt = f"""
                USER REQUEST: {current_input}
                
                FILE INFO:
                {file_info}
                
                Generate Python code to filter the data as requested. The filtered DataFrame should be assigned to 'filtered_df'.
                """
                
                response = openai_handler.get_response(prompt, system_message)
                
                # Extract code from the response
                code_block = extract_code(response)
                
                if code_block:
                    # Execute the filtering code
                    result = CodeExecutor.execute_code(code_block, df_to_export)
                    
                    # Check if filtered_df was created
                    if result['success'] and 'filtered_df' in result.get('local_vars', {}):
                        st.session_state.filtered_df = result['local_vars']['filtered_df']
                    else:
                        # If no filtered_df was created, use the original DataFrame
                        st.session_state.filtered_df = df_to_export
            else:
                # No filtering, export the entire DataFrame
                st.session_state.filtered_df = df_to_export
            
            # Show export input
            st.session_state.show_export_input = True
            
            # Add AI response
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "Please enter the path where you want to export the file and select the format."
            })
            
            return True
        else:
            # No file loaded
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "No file is currently loaded. Please load a file first before trying to export."
            })
            
            return True
    
    # Check if the user is asking a question about the processed PDF
    elif is_asking_about_pdf(current_input) and st.session_state.pdf_processed:
        # Show a spinner while processing
        with st.spinner("Searching the PDF for answers..."):
            # Answer the question using the RAG system
            result = st.session_state.rag_system.answer_question(current_input)
            
            # Format the answer with citations
            if result["sources"]:
                answer = f"{result['answer']}\n\n**Sources:**\n"
                for i, source in enumerate(result["sources"], 1):
                    answer += f"{i}. {source}\n"
            else:
                answer = result["answer"]
            
            # Add the answer to conversation
            st.session_state.conversation.append({
                "role": "assistant",
                "content": answer
            })
        
        return True
    
    # Check if user is asking for data analysis and a file is loaded
    elif (st.session_state.file_path and st.session_state.file_type in ['csv', 'excel', 'parquet']) or \
         (st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files):
        
        # Determine which DataFrame to use
        if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
            df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
            file_name = st.session_state.uploaded_files[st.session_state.active_file_id]['name']
        else:
            df = st.session_state.file_content
            file_name = st.session_state.file_path
        
        # Generate context for the AI
        file_info_dict = FileHandler.get_file_info(df)
        file_info = f"""
        File: {file_name}
        DataFrame shape: {df.shape}
        Columns: {list(df.columns)}
        
        Column types:
        {file_info_dict['column_types']}
        
        Missing values:
        {file_info_dict['missing_values']}
        """
        
        # Add context about other loaded files
        if len(st.session_state.uploaded_files) > 1:
            file_info += "\nOther available files:\n"
            for file_id, file_data in st.session_state.uploaded_files.items():
                if file_id != st.session_state.active_file_id:
                    file_info += f"- {file_data['name']} (shape: {file_data['content'].shape})\n"
        
        # Add context about the annual report if available
        annual_report_context = ""
        if st.session_state.annual_report_path and st.session_state.comparison_company:
            annual_report_context = f"""
            The user also has the annual report for {st.session_state.comparison_company} available at '{st.session_state.annual_report_path}'.
            They may want to compare the loaded data with information from this annual report.
            """
        
        # Updated system message to include descriptive output requirements
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
        4. ALWAYS include descriptive output about the DataFrame, including:
           - Print df.shape to show dimensions
           - Print df.columns.tolist() to show column names
           - Print df.head() to show the first few rows
           - Print df.describe() for numerical columns
           - Print df.info() to show data types and non-null values
        5. ALWAYS include plt.show() when creating visualizations
        6. For complex analysis, print intermediate results to make the process clear
        7. Print conclusions or insights from the data at the end

        Format your response with only the Python code inside triple backticks:
        ```python
        # Your code here
        ```
        """
        
        # Get Python code from AI
        response = openai_handler.get_response(
            prompt=f"""
            USER REQUEST: {current_input + annual_report_context}
            
            FILE INFO:
            {file_info}
            
            FILE PREVIEW:
            {file_info_dict['preview']}
            
            Based on this information, generate Python code to analyze the data as requested.
            The code should be clean, efficient, and focus on producing the requested visualization or analysis with descriptive output.
            """,
            system_message=system_message,
            model="gpt-4"
        )
        
        # Store the full response in the conversation history
        st.session_state.conversation.append({
            "role": "assistant",
            "content": response
        })
        
        # Extract code from the response
        code_block = extract_code(response)
        
        if code_block:
            # Execute the code silently
            result = CodeExecutor.execute_code(code_block, df)
            st.session_state.code_execution_result = result
            
            # If there were errors, add them to the response
            if result['errors'] and not result['success']:
                error_message = f"⚠️ There was an error executing the analysis: {result['errors']}"
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": error_message
                })
        
        return True
    else:
        # For general questions, get response from OpenAI
        prompt = f"User: {current_input}\nProvide a helpful response. If the user is asking about data analysis, suggest they load a CSV file first."
        
        # Include info about loaded files
        if st.session_state.uploaded_files:
            prompt += "\n\nCurrently loaded files:"
            for file_id, file_info in st.session_state.uploaded_files.items():
                prompt += f"\n- {file_info['name']} (shape: {file_info['content'].shape})"
        
        # Include info about annual report if available
        if st.session_state.annual_report_path and st.session_state.comparison_company:
            prompt += f"\n\nNote: The user has loaded an annual report for {st.session_state.comparison_company}."
        
        if st.session_state.pdf_processed:
            prompt += f"\n\nNote: The user has a processed PDF that they can ask questions about."
        
        response = openai_handler.get_response(prompt)
        
        # Add AI response to conversation
        st.session_state.conversation.append({
            "role": "assistant",
            "content": response
        })
        
        return True

# Function to extract company name from user input
def extract_company_name(text):
    # This is a simple extraction - could be enhanced with NLP techniques
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

# Main UI
st.title("🤖 Agentic AI Data Analyst")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enter your OpenAI API key to enable AI functionality"
    )
    
    if api_key:
        st.session_state.openai_api_key = api_key
        openai_handler.set_api_key(api_key)
    
    st.divider()
    
    # File information for multiple files
    if st.session_state.uploaded_files:
        st.subheader("Loaded Files")
        # Display all loaded files with radio buttons to select active file
        file_options = {file_id: file_info['name'] for file_id, file_info in st.session_state.uploaded_files.items()}
        
        selected_file_id = st.radio(
            "Select active file:",
            options=list(file_options.keys()),
            format_func=lambda x: file_options[x],
            key="file_selector"
        )
        
        # Set the active file
        if selected_file_id != st.session_state.active_file_id:
            st.session_state.active_file_id = selected_file_id
            # Also update the traditional file content for backward compatibility
            st.session_state.file_content = st.session_state.uploaded_files[selected_file_id]['content']
            st.session_state.file_path = st.session_state.uploaded_files[selected_file_id]['name']
            st.session_state.file_type = st.session_state.uploaded_files[selected_file_id]['type']
        
        if st.button("Clear all files", key="clear_all_files"):
            st.session_state.uploaded_files = {}
            st.session_state.active_file_id = None
            st.session_state.file_path = None
            st.session_state.file_content = None
            st.session_state.file_type = None
            st.rerun()
    else:
        st.warning("No files loaded")
    
    # Annual report information
    if st.session_state.annual_report_path:
        st.success(f"Annual report loaded: {os.path.basename(st.session_state.annual_report_path)}")
        if st.button("Clear report", key="clear_report"):
            st.session_state.annual_report_path = None
            st.session_state.comparison_company = None
            st.rerun()
    
    # Clear chat button in sidebar
    if st.button("Clear Chat History", key="clear_chat_button"):
        st.session_state.conversation = []
        st.success("Chat history has been cleared.")
        st.rerun()
    
    # About section
    st.divider()
    st.markdown("""
    ### About
    This app uses AI to analyze data and generate visualizations based on your requests.
    
    1. Chat with the AI
    2. Ask to load CSV/Excel/Parquet files (supports multiple files)
    3. Request data analysis
    4. Perform DataFrame operations (merge, join, concat, etc.)
    5. Compare with company annual reports
    6. Download PDFs directly via URL
    7. Ask questions about downloaded PDFs
    8. View the results
    9. Check data quality
    
    #### Multiple File Support
    You can upload multiple files and perform operations between them:
    - Merge datasets
    - Concatenate files
    - Join related data
    - Compare different files
    
    #### DataFrame Operations
    Type "perform dataframe operations" to access:
    - Merge/Join
    - Concatenate
    - Fill/Drop NA values
    - Melt/Pivot
    - Group By and Aggregate
    
    #### PDF Question Answering
    When you download a PDF, the app automatically processes it 
    for question answering. You can then ask questions about the
    content of the PDF and get AI-generated answers based on the
    information in the document.
    
    #### Data Quality Check
    Type "check data quality" to get a comprehensive report about
    your loaded data, including missing values, phone number validation,
    and other quality metrics.
    """)

# Main content area with two columns
col1, col2 = st.columns([3, 2])

# Left column (Chat interface)
with col1:
    st.header("Chat with AI")
    
    # Display conversation history
    for message in st.session_state.conversation:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.write(f"**You**: {content}")
        else:
            # Clean the AI response to hide code blocks
            if "```" in content:
                content = clean_response(content)
            st.write(f"**AI**: {content}")
    
    # File input section (shows only when needed)
    if st.session_state.show_file_input:
        st.subheader("Upload Files")
        
        uploaded_files = st.file_uploader(
            "Choose one or more files", 
            type=['csv', 'xlsx', 'xls', 'parquet'], 
            key="file_uploader",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            new_files_added = False
            
            for uploaded_file in uploaded_files:
                # Generate a unique ID for this file
                file_id = str(uuid.uuid4())
                
                try:
                    # Determine file type from the uploaded file
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'csv':
                        content = pd.read_csv(uploaded_file)
                        file_type = 'csv'
                    elif file_extension in ['xlsx', 'xls']:
                        content = pd.read_excel(uploaded_file)
                        file_type = 'excel'
                    elif file_extension == 'parquet':
                        content = pd.read_parquet(uploaded_file)
                        file_type = 'parquet'
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        continue
                    
                    # Add to uploaded files dictionary
                    st.session_state.uploaded_files[file_id] = {
                        'name': uploaded_file.name,
                        'content': content,
                        'type': file_type
                    }
                    
                    # Set as active file if it's the first one
                    if st.session_state.active_file_id is None:
                        st.session_state.active_file_id = file_id
                        # Also set the traditional file variables for backward compatibility
                        st.session_state.file_path = uploaded_file.name
                        st.session_state.file_content = content
                        st.session_state.file_type = file_type
                    
                    new_files_added = True
                        
                except Exception as e:
                    st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
            
            if new_files_added:
                # Add system message to conversation
                file_names = [info['name'] for info in st.session_state.uploaded_files.values()]
                
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Files loaded successfully: {', '.join(file_names)}. You can now ask me to analyze the data or perform DataFrame operations between files."
                })
                
                # Hide file input after successful load
                st.session_state.show_file_input = False
                st.rerun()

    # DataFrame Operations section (shows only when needed)
    if st.session_state.show_df_operations:
        st.subheader("DataFrame Operations")
        
        if len(st.session_state.uploaded_files) > 0:
            # Create multiselect for selecting files
            file_options = {file_id: file_info['name'] for file_id, file_info in st.session_state.uploaded_files.items()}
            
            selected_file_ids = st.multiselect(
                "Select files to operate on:",
                options=list(file_options.keys()),
                format_func=lambda x: file_options[x],
                default=[st.session_state.active_file_id] if st.session_state.active_file_id else [],
                key="operation_file_selector"
            )
            
            # Select operation type
            operation_type = st.selectbox(
                "Select operation type:",
                options=["merge", "concat", "fillna", "dropna", "melt", "pivot", "groupby"],
                key="operation_type"
            )
            
            # Parameters for each operation type
            params = {}
            
            if operation_type == "merge":
                if len(selected_file_ids) < 2:
                    st.warning("Merge operation requires at least 2 files. Please select another file.")
                else:
                    # Create a more user-friendly merge UI
                    
                    # Get metadata for the files
                    file1_name = st.session_state.uploaded_files[selected_file_ids[0]]['name']
                    file2_name = st.session_state.uploaded_files[selected_file_ids[1]]['name']
                    
                    # Get dataframes for both files
                    df1 = st.session_state.uploaded_files[selected_file_ids[0]]['content']
                    df2 = st.session_state.uploaded_files[selected_file_ids[1]]['content']
                    
                    # Get column lists
                    df1_columns = list(df1.columns)
                    df2_columns = list(df2.columns)
                    
                    # Find common columns for join key suggestions
                    common_columns = list(set(df1_columns) & set(df2_columns))
                    
                    # Default to "ID" column if it exists, otherwise use first common column
                    default_id_col = None
                    for col_name in ['ID', 'Id', 'id', 'iD']:
                        if col_name in common_columns:
                            default_id_col = col_name
                            break
                    
                    if default_id_col is None and common_columns:
                        default_id_col = common_columns[0]
                    elif default_id_col is None:
                        default_id_col = df1_columns[0] if df1_columns else None
                    
                    # Let user choose between simple column fetch or full join
                    operation_subtype = st.radio(
                        "Choose operation type:",
                        options=["Simple column fetch (add columns from second file)", 
                                "Full join (SQL-style merge with all options)"],
                        index=0,
                        key="operation_subtype"
                    )
                    
                    if operation_subtype == "Simple column fetch (add columns from second file)":
                        # Simple column fetch UI
                        st.write(f"### Add columns from {file2_name} to {file1_name}")
                        
                        # First: Ask what ID column to match on in each file
                        st.write("**Step 1:** Select the ID columns to match")
                        
                        col1a, col1b = st.columns(2)
                        with col1a:
                            params['left_on'] = st.selectbox(
                                f"ID column in {file1_name}:",
                                options=df1_columns,
                                index=df1_columns.index(default_id_col) if default_id_col in df1_columns else 0,
                                key="left_on"
                            )
                        
                        with col1b:
                            params['right_on'] = st.selectbox(
                                f"ID column in {file2_name}:",
                                options=df2_columns,
                                index=df2_columns.index(default_id_col) if default_id_col in df2_columns else 0,
                                key="right_on"
                            )
                        
                        # Check for duplicate keys in both dataframes
                        left_dupes = df1[params['left_on']].duplicated().any()
                        right_dupes = df2[params['right_on']].duplicated().any()
                        
                        if left_dupes or right_dupes:
                            st.warning(f"⚠️ Warning: Duplicate IDs detected! " + 
                                    (f"'{params['left_on']}' has duplicates in {file1_name}. " if left_dupes else "") +
                                    (f"'{params['right_on']}' has duplicates in {file2_name}. " if right_dupes else "") +
                                    "This may cause row duplication in the result.")
                            
                            # Offer option to handle duplicates
                            handle_dupes = st.selectbox(
                                "How to handle duplicates:",
                                options=["Keep first occurrence only", "Keep last occurrence only", "Keep all (may duplicate rows)"],
                                index=0,
                                key="handle_dupes"
                            )
                            params['handle_dupes'] = handle_dupes
                        
                        # Second: Ask what columns to fetch from the second file
                        st.write("**Step 2:** Select columns to fetch from the second table")
                        
                        # Exclude the join column from the fetch options to avoid duplicates
                        fetch_options = [col for col in df2_columns if col != params['right_on']]
                        
                        # Try to auto-detect "House number" or similar columns
                        default_fetch_cols = []
                        for col in fetch_options:
                            if col.lower().replace(" ", "") in ['housenumber', 'house_number', 'houseno', 'house']:
                                default_fetch_cols.append(col)
                        
                        # Allow selecting columns to fetch from second table
                        params['fetch_columns'] = st.multiselect(
                            f"Select columns to fetch from {file2_name}:",
                            options=fetch_options,
                            default=default_fetch_cols,
                            key="fetch_columns"
                        )
                        
                        # If nothing selected, offer to fetch all remaining columns
                        if not params['fetch_columns']:
                            fetch_all = st.checkbox("Fetch all columns", value=True, key="fetch_all")
                            if fetch_all:
                                params['fetch_columns'] = fetch_options
                        
                        # What to do with non-matching rows
                        params['how'] = st.radio(
                            "How to handle rows that don't have a match:",
                            options=["Keep rows with missing values", "Only keep rows with matches"],
                            index=0,
                            key="missing_handling",
                            help="'Keep rows with missing values' is like LEFT JOIN, 'Only keep rows with matches' is like INNER JOIN"
                        )
                        
                        # Convert the user-friendly options to the actual join type
                        if params['how'] == "Keep rows with missing values":
                            params['how'] = "left"
                        else:
                            params['how'] = "inner"
                        
                        # Simple mode is just column fetch
                        params['simple_mode'] = True
                        
                    else:
                        # Full join UI (original functionality)
                        st.write(f"### Full Join Operation: {file1_name} with {file2_name}")
                        
                        # Join key selection
                        col1a, col1b = st.columns(2)
                        with col1a:
                            params['left_on'] = st.selectbox(
                                f"Left join key column ({file1_name}):",
                                options=df1_columns,
                                index=df1_columns.index(default_id_col) if default_id_col in df1_columns else 0,
                                key="left_on"
                            )
                        
                        with col1b:
                            # If we have common columns, default to the same column name
                            # Otherwise, let the user pick any column from the right dataframe
                            right_default_index = df2_columns.index(params['left_on']) if params['left_on'] in df2_columns else 0
                            params['right_on'] = st.selectbox(
                                f"Right join key column ({file2_name}):",
                                options=df2_columns,
                                index=right_default_index,
                                key="right_on"
                            )
                        
                        # Select join type
                        params['how'] = st.selectbox(
                            "Join type:",
                            options=["left", "inner", "right", "outer"],
                            index=0,  # Default to left join
                            key="join_type"
                        )
                        
                        # Allow the user to select which columns to include from the right dataframe
                        st.write("Select columns to include from the right dataframe:")
                        
                        # Add 'All columns' option
                        all_cols_option = "[All columns]"
                        right_cols_options = [all_cols_option] + df2_columns
                        
                        selected_right_cols = st.multiselect(
                            "Columns to include (leave empty for all):",
                            options=right_cols_options,
                            default=[all_cols_option],
                            key="right_cols_to_include"
                        )
                        
                        # Store in params
                        if not selected_right_cols or all_cols_option in selected_right_cols:
                            params['right_cols'] = None  # All columns
                        else:
                            params['right_cols'] = selected_right_cols
                        
                        # Full mode - not just column fetch
                        params['simple_mode'] = False
            
            elif operation_type == "concat":
                params['axis'] = st.radio(
                    "Concatenation axis:",
                    options=[0, 1],
                    format_func=lambda x: "Rows (0)" if x == 0 else "Columns (1)",
                    key="concat_axis"
                )
                
                params['ignore_index'] = st.checkbox(
                    "Ignore index?",
                    value=True,
                    key="ignore_index"
                )
            
            elif operation_type == "fillna":
                method_options = ["value", "ffill", "bfill"]
                
                fill_method = st.radio(
                    "Fill method:",
                    options=method_options,
                    key="fill_method"
                )
                
                if fill_method == "value":
                    params['value'] = st.text_input(
                        "Fill value (use 0 for numeric, empty string for text):",
                        "0",
                        key="fill_value"
                    )
                    # Convert to appropriate type (int, float, or keep as string)
                    try:
                        if params['value'].isdigit():
                            params['value'] = int(params['value'])
                        elif params['value'].replace('.', '', 1).isdigit():
                            params['value'] = float(params['value'])
                    except:
                        pass
                else:
                    params['method'] = fill_method
            
            elif operation_type == "dropna":
                params['axis'] = st.radio(
                    "Drop axis:",
                    options=[0, 1],
                    format_func=lambda x: "Rows (0)" if x == 0 else "Columns (1)",
                    key="drop_axis"
                )
                
                params['how'] = st.radio(
                    "Drop condition:",
                    options=["any", "all"],
                    format_func=lambda x: f"Any {params['axis']}" if x == "any" else f"All {params['axis']}",
                    key="drop_how"
                )
            
            elif operation_type == "melt":
                if len(selected_file_ids) >= 1:
                    df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
                    columns = list(df.columns)
                    
                    params['id_vars'] = st.multiselect(
                        "ID Variables (columns to keep as is):",
                        options=columns,
                        key="id_vars"
                    )
                    
                    remaining_cols = [col for col in columns if col not in params['id_vars']]
                    
                    params['value_vars'] = st.multiselect(
                        "Value Variables (columns to melt):",
                        options=remaining_cols,
                        default=remaining_cols,
                        key="value_vars"
                    )
                    
                    col1a, col1b = st.columns(2)
                    with col1a:
                        params['var_name'] = st.text_input(
                            "Variable column name:",
                            "variable",
                            key="var_name"
                        )
                    
                    with col1b:
                        params['value_name'] = st.text_input(
                            "Value column name:",
                            "value",
                            key="value_name"
                        )
            
            elif operation_type == "pivot":
                if len(selected_file_ids) >= 1:
                    df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
                    columns = list(df.columns)
                    
                    col1a, col1b, col1c = st.columns(3)
                    with col1a:
                        params['index'] = st.selectbox(
                            "Index column:",
                            options=columns,
                            key="pivot_index"
                        )
                    
                    with col1b:
                        params['columns'] = st.selectbox(
                            "Columns to pivot:",
                            options=[col for col in columns if col != params.get('index')],
                            key="pivot_columns"
                        )
                    
                    with col1c:
                        params['values'] = st.selectbox(
                            "Values column (optional):",
                            options=[None] + [col for col in columns if col != params.get('index') and col != params.get('columns')],
                            key="pivot_values"
                        )
            
            elif operation_type == "groupby":
                if len(selected_file_ids) >= 1:
                    df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
                    columns = list(df.columns)
                    
                    params['by'] = st.multiselect(
                        "Group by columns:",
                        options=columns,
                        key="groupby_cols"
                    )
                    
                    # Only show numeric columns for aggregation
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    agg_functions = ["mean", "sum", "count", "min", "max", "std"]
                    agg_selections = {}
                    
                    st.write("Select aggregation functions for numeric columns:")
                    
                    agg_columns = st.multiselect(
                        "Select columns to aggregate:",
                        options=numeric_cols,
                        key="agg_cols"
                    )
                    
                    for col in agg_columns:
                        agg_selections[col] = st.multiselect(
                            f"Aggregations for {col}:",
                            options=agg_functions,
                            default=["mean"],
                            key=f"agg_func_{col}"
                        )
                    
                    # Convert selections to agg dict
                    params['agg'] = {col: funcs for col, funcs in agg_selections.items()}
            
            # Execute operation button
            if st.button("Execute Operation", key="execute_operation_button"):
                if selected_file_ids:
                    with st.spinner(f"Performing {operation_type} operation..."):
                        result_df, description = perform_dataframe_operation(
                            operation_type, 
                            params, 
                            selected_file_ids
                        )
                        
                        if result_df is not None:
                            # Store result
                            st.session_state.operation_result = result_df
                            st.session_state.operation_description = description
                            
                            # Create a new file entry for the result
                            result_id = str(uuid.uuid4())
                            st.session_state.uploaded_files[result_id] = {
                                'name': f"Result_{operation_type}_{len(st.session_state.uploaded_files)}",
                                'content': result_df,
                                'type': 'result'
                            }
                            
                            # Set as active file
                            st.session_state.active_file_id = result_id
                            st.session_state.file_content = result_df
                            st.session_state.file_path = st.session_state.uploaded_files[result_id]['name']
                            st.session_state.file_type = 'result'
                            
                            # Add to conversation
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": f"I've performed the {operation_type} operation successfully. Here's a summary of the result:\n\n{description}"
                            })
                            
                            # Hide operations UI
                            st.session_state.show_df_operations = False
                            st.rerun()
                        else:
                            st.error(f"Operation failed: {description}")
                else:
                    st.error("Please select at least one file for the operation.")
        else:
            st.warning("No files loaded. Please upload files first.")
    
    # PDF URL input section (shows only when needed)
    if st.session_state.show_pdf_url_input:
        st.subheader("Enter PDF URL")
        pdf_url = st.text_input("URL of the PDF file:", key="pdf_url_input")
        custom_filename = st.text_input("Custom filename (optional):", key="custom_filename_input", 
                                       help="Leave blank to use automatically generated filename")
        
        if st.button("Download PDF", key="download_pdf_button"):
            if pdf_url:
                try:
                    # Show downloading message with spinner
                    with st.spinner(f"Downloading PDF from {pdf_url}..."):
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": f"Starting download from {pdf_url}..."
                        })
                        
                        # Download the PDF with a timeout
                        import time
                        start_time = time.time()
                        
                        # Download the PDF
                        filepath, message = FileHandler.download_pdf_from_link(
                            pdf_url, 
                            filename=custom_filename if custom_filename else None
                        )
                        
                        download_time = time.time() - start_time
                        
                        if filepath:
                            # PDF downloaded successfully
                            st.session_state.annual_report_path = filepath
                            
                            # Set comparison company name if we're doing a comparison
                            if st.session_state.pdf_download_purpose == "comparison":
                                # Try to extract from previous messages or use filepath
                                if st.session_state.comparison_company:
                                    # Already set from earlier in the conversation
                                    pass
                                else:
                                    for msg in reversed(st.session_state.conversation):
                                        if msg["role"] == "user":
                                            extracted_name = extract_company_name(msg["content"])
                                            if extracted_name:
                                                st.session_state.comparison_company = extracted_name
                                                break
                                    
                                    # If still not set, use filename
                                    if not st.session_state.comparison_company:
                                        # Extract from filename
                                        base_name = os.path.basename(filepath)
                                        st.session_state.comparison_company = base_name.split('_')[0].capitalize()
                            
                            # Initialize RAG system and process PDF
                            with st.spinner("Processing PDF for question answering..."):
                                # Init RAG system if needed
                                if not st.session_state.rag_system:
                                    st.session_state.rag_system = RAGSystem(st.session_state.openai_api_key)
                                
                                # Process the PDF
                                success, message = st.session_state.rag_system.process_pdf(filepath)
                                st.session_state.pdf_processed = success
                                
                                if success:
                                    process_message = "PDF processed successfully. You can now ask questions about the content of the PDF."
                                else:
                                    process_message = f"PDF processing failed: {message}"
                            
                            # Add success message to conversation
                            st.session_state.conversation[-1] = {
                                "role": "assistant",
                                "content": f"PDF file downloaded successfully to {os.path.basename(filepath)} in {download_time:.2f} seconds. {process_message}"
                            }
                        else:
                            # Failed to download PDF
                            st.session_state.conversation[-1] = {
                                "role": "assistant",
                                "content": f"Failed to download the PDF: {message}"
                            }
                    
                    # Hide URL input after attempt
                    st.session_state.show_pdf_url_input = False
                    st.session_state.pdf_download_purpose = None
                    st.rerun()
                    
                except Exception as e:
                    import traceback
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.text_area("Error details", value=traceback.format_exc(), height=150)
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"An error occurred while trying to download the PDF: {str(e)}. Please try again with a different URL."
                    })
            else:
                st.error("Please enter a URL")
    
    # Company input section (shows only when needed)
    if st.session_state.show_company_input:
        st.subheader("Enter Company Name")
        company_name = st.text_input("Company name:", key="company_name_input")
        
        # Add options to either find report or provide URL
        col1a, col1b = st.columns(2)
        
        with col1a:
            if st.button("Find Annual Report", key="find_report_button"):
                if company_name and st.session_state.openai_api_key:
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"Looking for the latest annual report for {company_name}. This may take a few moments..."
                    })
                    st.rerun()
                    
                    try:
                        annual_report_agent = AnnualReportAgent(st.session_state.openai_api_key)
                        result = annual_report_agent.find_and_download_annual_report(company_name)
                        
                        if result and os.path.exists(result):
                            st.session_state.annual_report_path = result
                            st.session_state.comparison_company = company_name
                            
                            # Initialize RAG system and process PDF
                            with st.spinner("Processing PDF for question answering..."):
                                # Init RAG system if needed
                                if not st.session_state.rag_system:
                                    st.session_state.rag_system = RAGSystem(st.session_state.openai_api_key)
                                
                                # Process the PDF
                                success, message = st.session_state.rag_system.process_pdf(result)
                                st.session_state.pdf_processed = success
                                
                                if success:
                                    process_message = "PDF processed successfully. You can now ask questions about the content of the PDF."
                                else:
                                    process_message = f"PDF processing failed: {message}"
                            
                            # Add system message to conversation
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": f"Annual report for {company_name} downloaded successfully. {process_message}"
                            })
                        else:
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": f"I couldn't find or download the annual report for {company_name}. Would you like to provide a direct URL to the PDF file instead?"
                            })
                            # Show PDF URL input instead
                            st.session_state.show_pdf_url_input = True
                            st.session_state.pdf_download_purpose = "comparison"
                    except Exception as e:
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": f"An error occurred: {str(e)}"
                        })
                    
                    # Hide company input after attempt
                    st.session_state.show_company_input = False
                    st.rerun()
                else:
                    if not company_name:
                        st.error("Please enter a company name")
                    if not st.session_state.openai_api_key:
                        st.error("Please enter your OpenAI API key in the sidebar")
        
        with col1b:
            if st.button("Provide PDF URL Instead", key="provide_pdf_url_button"):
                st.session_state.show_company_input = False
                st.session_state.show_pdf_url_input = True
                st.session_state.pdf_download_purpose = "comparison"
                if company_name:
                    st.session_state.comparison_company = company_name
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Please provide the direct URL to {company_name if company_name else 'the company'}'s annual report PDF."
                })
                st.rerun()
    
    # File export section (shows only when needed)
    if st.session_state.show_export_input:
        st.subheader("Export Data")
        
        # Container for export options
        export_container = st.container()
        with export_container:
            # Get export path
            export_path = st.text_input("Enter export path (including filename):", key="export_path_input")
            
            # Select export format
            export_format = st.selectbox(
                "Select export format:",
                options=["CSV", "XLSX (Excel)", "Parquet"],
                index=0,
                key="export_format"
            )
            
            # Map the display format to actual format
            format_mapping = {
                "CSV": "csv",
                "XLSX (Excel)": "xlsx",
                "Parquet": "parquet"
            }
            
            # Export button
            if st.button("Export File", key="export_file_button"):
                if export_path:
                    # Determine which DataFrame to export
                    df_to_export = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.file_content
                    
                    if df_to_export is not None and isinstance(df_to_export, pd.DataFrame):
                        # Export the DataFrame using the mapped format
                        message, success = FileHandler.export_dataframe(
                            df_to_export, 
                            export_path, 
                            format_mapping[export_format]
                        )
                        
                        if success:
                            # Add success message to conversation
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": message
                            })
                            
                            # Hide export input
                            st.session_state.show_export_input = False
                            st.session_state.filtered_df = None
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("No data available to export.")
                else:
                    st.error("Please enter an export path.")
    
    # Create a form to handle message submission properly
    with st.form(key="message_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button:
            if not user_input.strip():
                st.warning("Please enter a message")
            elif not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
            else:
                # Store the message
                current_input = user_input
                
                # Add user message to conversation
                st.session_state.conversation.append({
                    "role": "user",
                    "content": current_input
                })
                
                # Process the message
                should_rerun = process_message(current_input)
                
                # Handle comparison follow-up buttons if needed
                if is_asking_for_comparison(current_input):
                    company_name = extract_company_name(current_input)
                    st.rerun()
                
                # Always rerun to refresh the page and show results
                if should_rerun:
                    st.rerun()
    
    # Handle comparison follow-up buttons if the latest message is about comparison
    if st.session_state.conversation and len(st.session_state.conversation) >= 2:
        last_user_msg = ""
        for msg in reversed(st.session_state.conversation):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if is_asking_for_comparison(last_user_msg):
            company_name = extract_company_name(last_user_msg)
            
            # Add buttons for the two options
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("Find automatically", key="auto_find"):
                    st.session_state.show_company_input = True
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"Please confirm the company name or enter a different one."
                    })
                    st.rerun()
            with col1b:
                if st.button("Provide PDF URL", key="provide_url"):
                    st.session_state.show_pdf_url_input = True
                    st.session_state.pdf_download_purpose = "comparison"
                    if company_name:
                        st.session_state.comparison_company = company_name
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"Please provide the direct URL to {company_name if company_name else 'the company'}'s annual report PDF."
                    })
                    st.rerun()

# Right column (Data & Analysis)
with col2:
    st.header("Data & Analysis")
    
    # Display active file preview if available
    if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
        file_info = st.session_state.uploaded_files[st.session_state.active_file_id]
        
        st.subheader(f"Data Preview: {file_info['name']}")
        st.dataframe(file_info['content'].head(10))
        
        # Add a button to check data quality directly
        if st.button("Check Data Quality", key="check_data_quality"):
            quality_report = generate_data_quality_report(file_info['content'])
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"# Data Quality Report for {file_info['name']}\n\n{quality_report}"
            })
            st.rerun()
        
        # Display metadata
        with st.expander("Dataset Information"):
            st.text(f"Rows: {file_info['content'].shape[0]}")
            st.text(f"Columns: {file_info['content'].shape[1]}")
            st.text("Data Types:")
            st.write(file_info['content'].dtypes)
            
            # Show DataFrame operations button
            if st.button("Perform DataFrame Operations", key="perform_df_operations"):
                st.session_state.show_df_operations = True
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": "Please select the DataFrame operation you'd like to perform using the panel below."
                })
                st.rerun()
    
    # Display operation result if available
    elif st.session_state.operation_result is not None:
        st.subheader("Operation Result")
        st.dataframe(st.session_state.operation_result.head(10))
        
        # Display operation details
        with st.expander("Operation Details"):
            st.markdown(st.session_state.operation_description)
    
    # Display file preview from the old way if available
    elif st.session_state.file_path and st.session_state.file_type in ['csv', 'excel', 'parquet']:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.file_content.head(10))
        
        # Add a button to check data quality directly
        if st.button("Check Data Quality", key="check_data_quality"):
            quality_report = generate_data_quality_report(st.session_state.file_content)
            st.session_state.conversation.append({
                "role": "assistant",
                "content": quality_report
            })
            st.rerun()
        
        # Display metadata
        with st.expander("Dataset Information"):
            st.text(f"Rows: {st.session_state.file_content.shape[0]}")
            st.text(f"Columns: {st.session_state.file_content.shape[1]}")
            st.text("Data Types:")
            st.write(st.session_state.file_content.dtypes)
    
    # Display annual report info if available
    if st.session_state.annual_report_path:
        st.subheader(f"{st.session_state.comparison_company or 'Downloaded'} Report")
        
        # Show different status based on processing state
        if st.session_state.pdf_processed:
            st.success(f"Report loaded and processed: {os.path.basename(st.session_state.annual_report_path)}")
            st.info("You can now ask questions about the content of this PDF.")
        else:
            st.info(f"Report loaded: {os.path.basename(st.session_state.annual_report_path)}")
            if not st.session_state.pdf_processed and st.session_state.rag_system:
                if st.button("Process PDF for Q&A", key="process_pdf"):
                    with st.spinner("Processing PDF for question answering..."):
                        success, message = st.session_state.rag_system.process_pdf(st.session_state.annual_report_path)
                        st.session_state.pdf_processed = success
                        if success:
                            st.success("PDF processed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to process PDF: {message}")
        
        # Add a button to open the PDF
        if st.button("Open PDF", key="open_pdf"):
            import webbrowser
            webbrowser.open_new_tab(f"file://{os.path.abspath(st.session_state.annual_report_path)}")
        
        # If processed, add a button to ask a question about the PDF
        if st.session_state.pdf_processed:
            pdf_question = st.text_input("Ask a question about the PDF:", key="pdf_question_input")
            if st.button("Ask", key="ask_pdf_question") and pdf_question:
                st.session_state.conversation.append({
                    "role": "user",
                    "content": pdf_question
                })
                
                with st.spinner("Searching the PDF for answers..."):
                    # Answer the question using the RAG system
                    result = st.session_state.rag_system.answer_question(pdf_question)
                    
                    # Format the answer with citations
                    if result["sources"]:
                        answer = f"{result['answer']}\n\n**Sources:**\n"
                        for i, source in enumerate(result["sources"], 1):
                            answer += f"{i}. {source}\n"
                    else:
                        answer = result["answer"]
                    
                    # Add the answer to conversation
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": answer
                    })
                
                st.rerun()
    
    # Display code execution results if available
    if st.session_state.code_execution_result:
        result = st.session_state.code_execution_result
        
        # Display figures (primary output)
        if result['figures']:
            st.subheader("Visualization Results")
            for i, fig in enumerate(result['figures']):
                st.pyplot(fig)
        
        # Display text output if there are no figures or if there is output
        if result['output']:
            st.subheader("Analysis Results")
            st.text_area("", value=result['output'], height=300, disabled=True)


