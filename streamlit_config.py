import streamlit as st
import pandas as pd
import uuid
import os
from typing import Dict, List, Tuple, Any, Optional, Callable
import re
from llm import MessageHandler, OpenAIHandler
from data_operation import DataFrameOperations, DataAnalyzer
from pdf_fetcher import PDFDownloader, AnnualReportAgent, PDFRequestParser
from rag_app import RAGSystem, DocumentQASession
from file_export import FileHandler, FileExporter
from python_engine import CodeExecutor


class AppSessionState:
    """
    Manages the application's session state.
    
    This class encapsulates the initialization and management of the
    application's session state variables, making it easier to track 
    conversation history, loaded files, active operations, and other
    stateful aspects of the application.
    """
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables if they don't exist."""
        # Conversation and user interaction
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
        
        # File handling states
        if "file_path" not in st.session_state:
            st.session_state.file_path = None
        if "file_content" not in st.session_state:
            st.session_state.file_content = None
        if "file_type" not in st.session_state:
            st.session_state.file_type = None
        
        # Multiple file support
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}  # dict to store multiple files {id: {name, content, type}}
        if "active_file_id" not in st.session_state:
            st.session_state.active_file_id = None  # currently selected file
        
        # API key state
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # Duplicate detection states
        if "duplicate_results" not in st.session_state:
            st.session_state.duplicate_results = None
        if "duplicate_summary" not in st.session_state:
            st.session_state.duplicate_summary = None
        if "show_duplicate_preview" not in st.session_state:
            st.session_state.show_duplicate_preview = False
        if "show_duplicate_settings" not in st.session_state:
            st.session_state.show_duplicate_settings = False

        # UI control states
        if "show_file_input" not in st.session_state:
            st.session_state.show_file_input = False
        if "show_export_input" not in st.session_state:
            st.session_state.show_export_input = False
        if "show_df_operations" not in st.session_state:
            st.session_state.show_df_operations = False
        if "show_company_input" not in st.session_state:
            st.session_state.show_company_input = False
        if "show_pdf_url_input" not in st.session_state:
            st.session_state.show_pdf_url_input = False
        
        # Data operation states
        if "filtered_df" not in st.session_state:
            st.session_state.filtered_df = None
        if "operation_result" not in st.session_state:
            st.session_state.operation_result = None
        if "operation_description" not in st.session_state:
            st.session_state.operation_description = None
        
        # PDF and report handling states
        if "annual_report_path" not in st.session_state:
            st.session_state.annual_report_path = None
        if "comparison_company" not in st.session_state:
            st.session_state.comparison_company = None
        if "pdf_download_purpose" not in st.session_state:
            st.session_state.pdf_download_purpose = None
        
        # RAG system states
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = None
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False
        if "pdf_chunks" not in st.session_state:
            st.session_state.pdf_chunks = None
        
        # Code execution results
        if "code_execution_result" not in st.session_state:
            st.session_state.code_execution_result = None
    
    @staticmethod
    def reset_conversation():
        """Reset the conversation history."""
        st.session_state.conversation = []
    
    @staticmethod
    def add_user_message(message: str):
        """Add a user message to the conversation history."""
        st.session_state.conversation.append({
            "role": "user",
            "content": message
        })
    
    @staticmethod
    def add_assistant_message(message: str):
        """Add an assistant message to the conversation history."""
        st.session_state.conversation.append({
            "role": "assistant",
            "content": message
        })
    
    @staticmethod
    def clear_ui_states():
        """Clear UI control states to reset the interface."""
        st.session_state.show_file_input = False
        st.session_state.show_export_input = False
        st.session_state.show_df_operations = False
        st.session_state.show_company_input = False
        st.session_state.show_pdf_url_input = False


class AppUIManager:
    """
    Manages the application's UI components and layout.
    
    This class encapsulates the creation and configuration of the app's
    UI components, including the sidebar, main content area, and specialized
    panels like file upload, dataframe operations, and PDF handling.
    """
    @staticmethod
    def configure_page():
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="Agentic AI Data Analyst",
            page_icon="🤖",
            layout="wide"
        )

    @staticmethod
    def render_duplicate_settings():
        """Render the duplicate detection settings interface."""
        if st.session_state.show_duplicate_settings:
            st.subheader("Duplicate Detection Settings")
            
            # Get active DataFrame
            df = None
            if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
            elif st.session_state.file_content is not None:
                df = st.session_state.file_content
            
            if df is not None:
                # Let user select name columns
                col1, col2 = st.columns(2)
                
                with col1:
                    first_name_col = st.selectbox(
                        "First Name Column:",
                        options=[col for col in df.columns if 'name' in col.lower() or 'first' in col.lower()],
                        key="first_name_col",
                        index=0 if any('first' in col.lower() for col in df.columns) else 0
                    )
                
                with col2:
                    last_name_col = st.selectbox(
                        "Last Name Column:",
                        options=[col for col in df.columns if 'name' in col.lower() or 'last' in col.lower()],
                        key="last_name_col",
                        index=0 if any('last' in col.lower() for col in df.columns) else 0
                    )
                
                # Let user select similarity threshold
                threshold = st.select_slider(
                    "Similarity Threshold:",
                    options=["low", "medium", "high", "exact"],
                    value="medium",
                    key="similarity_threshold"
                )
                
                # Explanation of thresholds
                threshold_explanations = {
                    "low": "More potential matches, higher chance of false positives",
                    "medium": "Balanced approach, good for most datasets",
                    "high": "Fewer matches, lower chance of false positives",
                    "exact": "Only exact matches, no fuzzy matching"
                }
                
                st.info(f"**{threshold.capitalize()}**: {threshold_explanations[threshold]}")
                
                # Execute button
                detect_button = st.button("Detect Duplicates", key="detect_duplicates_button")
                
                return first_name_col, last_name_col, threshold, detect_button
            else:
                st.warning("No data loaded. Please upload a file first.")
        
        return None, None, None, False
    
    @staticmethod
    def render_duplicate_preview():
        """Render the duplicate detection results preview."""
        if st.session_state.show_duplicate_preview and st.session_state.duplicate_results is not None:
            st.subheader("Duplicate Records")
            
            # Show summary metrics
            if st.session_state.duplicate_summary:
                summary = st.session_state.duplicate_summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Duplicate Groups", summary['duplicate_groups'])
                
                with col2:
                    st.metric("Total Duplicate Records", summary['total_duplicates'])
                
                with col3:
                    st.metric("Percentage of Dataset", f"{summary['percent_duplicates']}%")
            
            # Show duplicate records grouped by duplicate_group
            duplicate_df = st.session_state.duplicate_results
            
            # Show each group in an expander
            for group_id in sorted(duplicate_df['duplicate_group'].unique()):
                group_df = duplicate_df[duplicate_df['duplicate_group'] == group_id]
                
                with st.expander(f"Group {int(group_id)} ({len(group_df)} records)"):
                    st.dataframe(group_df)
            
            # Option to export results
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Duplicate Records", key="export_duplicates"):
                    st.session_state.filtered_df = duplicate_df
                    st.session_state.show_export_input = True
                    st.rerun()
            
            with col2:
                if st.button("Close Preview", key="close_duplicate_preview"):
                    st.session_state.show_duplicate_preview = False
                    st.rerun()
    
    @staticmethod
    def render_sidebar(openai_handler: OpenAIHandler):
        """
        Render the application sidebar.
        
        Args:
            openai_handler: OpenAI handler instance to update API key
        """
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
                st.warning("No CSV files loaded yet!")
            
            # Annual report information
            if st.session_state.annual_report_path:
                st.success(f"File loaded: {os.path.basename(st.session_state.annual_report_path)}")
                if st.button("Clear report", key="clear_report"):
                    st.session_state.annual_report_path = None
                    st.session_state.comparison_company = None
                    st.rerun()
            
            # Clear chat button in sidebar
            if st.button("Clear Chat History", key="clear_chat_button"):
                AppSessionState.reset_conversation()
                st.success("Chat history has been cleared.")
                st.rerun()
            
            # About section
            st.divider()
            AppUIManager.render_about_section()
    
    @staticmethod
    def render_about_section():
        """Render the about section with app information."""
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
        """)
    
    @staticmethod
    def render_chat_interface():
        """Render the chat interface with conversation history."""
        st.header("Agentic AI Data Analyst")
        
        # Display conversation history
        for message in st.session_state.conversation:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.write(f"**You**: {content}")
            else:
                # Clean the AI response to hide code blocks
                if "```" in content:
                    content = MessageHandler.clean_response(content)
                st.write(f"**AI**: {content}")
    
    @staticmethod
    def render_file_uploader():
        """Render the file upload interface."""
        if st.session_state.show_file_input:
            st.subheader("Upload Files")
            
            uploaded_files = st.file_uploader(
                "Choose one or more files", 
                type=['csv', 'xlsx', 'xls', 'parquet'], 
                key="file_uploader",
                accept_multiple_files=True
            )
            
            return uploaded_files
        return None
    
    @staticmethod
    def render_pdf_url_input():
        """Render the PDF URL input interface."""
        if st.session_state.show_pdf_url_input:
            st.subheader("Enter PDF URL")
            pdf_url = st.text_input("URL of the PDF file:", key="pdf_url_input")
            custom_filename = st.text_input("Custom filename (optional):", key="custom_filename_input", 
                                           help="Leave blank to use automatically generated filename")
            
            download_button = st.button("Download PDF", key="download_pdf_button")
            
            return pdf_url, custom_filename, download_button
        return None, None, False
    
    @staticmethod
    def render_company_input():
        """Render the company name input interface."""
        if st.session_state.show_company_input:
            st.subheader("Enter Company Name")
            company_name = st.text_input("Company name:", key="company_name_input")
            
            # Add options to either find report or provide URL
            col1a, col1b = st.columns(2)
            
            with col1a:
                find_report_button = st.button("Find Annual Report", key="find_report_button")
            
            with col1b:
                provide_url_button = st.button("Provide PDF URL Instead", key="provide_pdf_url_button")
            
            return company_name, find_report_button, provide_url_button
        return None, False, False
    
    @staticmethod
    def render_dataframe_operations():
        """Render the DataFrame operations interface."""
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
                
                # Create container for parameters based on operation type
                params_container = st.container()
                params = {}
                
                with params_container:
                    if operation_type == "merge":
                        params = AppUIManager._render_merge_params(selected_file_ids)
                    elif operation_type == "concat":
                        params = AppUIManager._render_concat_params()
                    elif operation_type == "fillna":
                        params = AppUIManager._render_fillna_params()
                    elif operation_type == "dropna":
                        params = AppUIManager._render_dropna_params()
                    elif operation_type == "melt":
                        params = AppUIManager._render_melt_params(selected_file_ids)
                    elif operation_type == "pivot":
                        params = AppUIManager._render_pivot_params(selected_file_ids)
                    elif operation_type == "groupby":
                        params = AppUIManager._render_groupby_params(selected_file_ids)
                
                # Execute operation button
                execute_button = st.button("Execute Operation", key="execute_operation_button")
                
                return selected_file_ids, operation_type, params, execute_button
            else:
                st.warning("No files loaded. Please upload files first.")
        
        return None, None, None, False
    
    @staticmethod
    def _render_merge_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for merge operation."""
        params = {}
        
        if len(selected_file_ids) < 2:
            st.warning("Merge operation requires at least 2 files. Please select another file.")
            return params
        
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
            
            # Allow selecting columns to fetch from second table
            params['fetch_columns'] = st.multiselect(
                f"Select columns to fetch from {file2_name}:",
                options=fetch_options,
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
            # Full join UI
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
        
        return params
    
    @staticmethod
    def _render_concat_params() -> Dict[str, Any]:
        """Render and collect parameters for concatenation operation."""
        params = {}
        
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
        
        return params
    
    @staticmethod
    def _render_fillna_params() -> Dict[str, Any]:
        """Render and collect parameters for fillna operation."""
        params = {}
        
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
        
        return params
    
    @staticmethod
    def _render_dropna_params() -> Dict[str, Any]:
        """Render and collect parameters for dropna operation."""
        params = {}
        
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
        
        return params
    
    @staticmethod
    def _render_melt_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for melt operation."""
        params = {}
        
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
        
        return params
    
    @staticmethod
    def _render_pivot_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for pivot operation."""
        params = {}
        
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
        
        return params
    
    @staticmethod
    def _render_groupby_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for groupby operation."""
        params = {}
        
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
        
        return params
    
    @staticmethod
    def render_message_form() -> Tuple[str, bool]:
        """
        Render the message input form.
        
        Returns:
            tuple: (user_input, submit_pressed)
        """
        with st.form(key="message_form", clear_on_submit=True):
            user_input = st.text_area("Your message:", height=100)
            submit_button = st.form_submit_button("Send")
            
            return user_input, submit_button
    
    @staticmethod
    def render_export_form():
        """Render the file export form."""
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
                export_button = st.button("Export File", key="export_file_button")
                
                return export_path, format_mapping[export_format], export_button
        
        return None, None, False
    
    @staticmethod
    def render_data_preview():
        """Render the data preview panel."""
        st.header("Data & Analysis")
        
        # Display active file preview if available
        if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
            file_info = st.session_state.uploaded_files[st.session_state.active_file_id]
            
            st.subheader(f"Data Preview: {file_info['name']}")
            st.dataframe(file_info['content'].head(10))
            
            # Add a button to check data quality directly
            check_quality = st.button("Check Data Quality", key="check_data_quality")
            
            # Display metadata
            with st.expander("Dataset Information"):
                st.text(f"Rows: {file_info['content'].shape[0]}")
                st.text(f"Columns: {file_info['content'].shape[1]}")
                st.text("Data Types:")
                st.write(file_info['content'].dtypes)
                
                # Show DataFrame operations button
                show_operations = st.button("Perform DataFrame Operations", key="perform_df_operations")
            
            return check_quality, show_operations
        
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
            check_quality = st.button("Check Data Quality", key="check_data_quality")
            
            # Display metadata
            with st.expander("Dataset Information"):
                st.text(f"Rows: {st.session_state.file_content.shape[0]}")
                st.text(f"Columns: {st.session_state.file_content.shape[1]}")
                st.text("Data Types:")
                st.write(st.session_state.file_content.dtypes)
            
            return check_quality, False
        
        return False, False
    
    @staticmethod
    def render_annual_report_info():
        """Render information about the loaded annual report."""
        if st.session_state.annual_report_path:
            st.subheader(f"{st.session_state.comparison_company or 'Downloaded'} Report")
            
            # Show different status based on processing state
            if st.session_state.pdf_processed:
                st.success(f"File loaded and processed: {os.path.basename(st.session_state.annual_report_path)}")
                st.info("You can now ask questions about the content of this PDF.")
            else:
                st.info(f"Report loaded: {os.path.basename(st.session_state.annual_report_path)}")
                if not st.session_state.pdf_processed and st.session_state.rag_system:
                    process_pdf_button = st.button("Process PDF for Q&A", key="process_pdf")
                    
                    if process_pdf_button:
                        return True
            
            # Add a button to open the PDF
            open_pdf_button = st.button("Open PDF", key="open_pdf")
            
            # If processed, add a button to ask a question about the PDF
            if st.session_state.pdf_processed:
                pdf_question = st.text_input("Ask a question about the PDF:", key="pdf_question_input")
                ask_pdf_button = st.button("Ask", key="ask_pdf_question")
                
                if ask_pdf_button and pdf_question:
                    return pdf_question
            
            if open_pdf_button:
                import webbrowser
                webbrowser.open_new_tab(f"file://{os.path.abspath(st.session_state.annual_report_path)}")
        
        return None
    
    @staticmethod
    def render_code_execution_results():
        """Render the results of code execution."""
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


class AppController:
    """
    Manages the application's control flow and business logic.
    
    This class orchestrates interactions between UI components and backend
    services, handling user requests, data processing, and state management.
    """
    def __init__(self):
        """Initialize the application controller."""
        AppSessionState.initialize_session_state()
        AppUIManager.configure_page()
        self.openai_handler = OpenAIHandler(st.session_state.openai_api_key)
    
    def handle_file_upload(self, uploaded_files):
        """
        Handle file upload process.
        
        Args:
            uploaded_files: Files uploaded through the Streamlit file uploader
        """
        if not uploaded_files:
            return
        
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
            
            AppSessionState.add_assistant_message(
                f"Files loaded successfully: {', '.join(file_names)}. You can now ask me to analyze the data or perform DataFrame operations between files."
            )
            
            # Hide file input after successful load
            st.session_state.show_file_input = False
            st.rerun()
    
    def handle_pdf_download(self, pdf_url, custom_filename):
        """
        Handle PDF download process.
        
        Args:
            pdf_url: URL of the PDF to download
            custom_filename: Optional custom filename for the downloaded PDF
        """
        if not pdf_url:
            st.error("Please enter a URL")
            return
        
        try:
            # Show downloading message with spinner
            with st.spinner(f"Downloading PDF from {pdf_url}..."):
                AppSessionState.add_assistant_message(f"Starting download from {pdf_url}...")
                
                # Download the PDF with a timeout
                import time
                start_time = time.time()
                
                # Download the PDF
                filepath, message = PDFDownloader.download_pdf_from_link(
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
                                    extracted_name = PDFRequestParser.extract_company_name(msg["content"])
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
                    
                    # Update the message
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
            AppSessionState.add_assistant_message(
                f"An error occurred while trying to download the PDF: {str(e)}. Please try again with a different URL."
            )
    
    def handle_company_report_request(self, company_name):
        """
        Handle request to find and download company annual report.
        
        Args:
            company_name: Name of the company
        """
        if not company_name:
            st.error("Please enter a company name")
            return
        
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
            return
        
        AppSessionState.add_assistant_message(
            f"Looking for the latest annual report for {company_name}. This may take a few moments..."
        )
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
                AppSessionState.add_assistant_message(
                    f"Annual report for {company_name} downloaded successfully. {process_message}"
                )
            else:
                AppSessionState.add_assistant_message(
                    f"I couldn't find or download the annual report for {company_name}. Would you like to provide a direct URL to the PDF file instead?"
                )
                # Show PDF URL input instead
                st.session_state.show_pdf_url_input = True
                st.session_state.pdf_download_purpose = "comparison"
        except Exception as e:
            AppSessionState.add_assistant_message(f"An error occurred: {str(e)}")
        
        # Hide company input after attempt
        st.session_state.show_company_input = False
        st.rerun()
    
    def handle_dataframe_operation(self, selected_file_ids, operation_type, params):
        """
        Handle DataFrame operation execution.
        
        Args:
            selected_file_ids: IDs of selected files
            operation_type: Type of operation to perform
            params: Parameters for the operation
        """
        if not selected_file_ids:
            st.error("Please select at least one file for the operation.")
            return
        
        with st.spinner(f"Performing {operation_type} operation..."):
            # Get the DataFrames to operate on
            dfs = []
            df_names = []
            
            for file_id in selected_file_ids:
                if file_id in st.session_state.uploaded_files:
                    file_info = st.session_state.uploaded_files[file_id]
                    dfs.append(file_info['content'])
                    df_names.append(file_info['name'])
            
            # Perform the operation
            result_df, description = DataFrameOperations.perform_operation(
                operation_type, 
                params, 
                dfs,
                df_names
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
                AppSessionState.add_assistant_message(
                    f"I've performed the {operation_type} operation successfully. Here's a summary of the result:\n\n{description}"
                )
                
                # Hide operations UI
                st.session_state.show_df_operations = False
                st.rerun()
            else:
                st.error(f"Operation failed: {description}")
    
    def handle_export_request(self, export_path, export_format):
        """
        Handle file export request.
        
        Args:
            export_path: Path where to save the file
            export_format: Format to use for export (csv, xlsx, parquet)
        """
        if not export_path:
            st.error("Please enter an export path.")
            return
        
        # Determine which DataFrame to export
        df_to_export = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.file_content
        
        if df_to_export is not None and isinstance(df_to_export, pd.DataFrame):
            # Export the DataFrame using the mapped format
            message, success = FileExporter.export_dataframe(
                df_to_export, 
                export_path, 
                export_format
            )
            
            if success:
                # Add success message to conversation
                AppSessionState.add_assistant_message(message)
                
                # Hide export input
                st.session_state.show_export_input = False
                st.session_state.filtered_df = None
                st.rerun()
            else:
                st.error(message)
        else:
            st.error("No data available to export.")
    
    def handle_data_quality_check(self, df: pd.DataFrame, file_name: str = None):
        """
        Handle data quality check request.
        
        Args:
            df: DataFrame to check
            file_name: Name of the file being checked
        """
        # Generate data quality report
        quality_report = DataAnalyzer.generate_data_quality_report(df)
        
        if file_name:
            report_title = f"# Data Quality Report for {file_name}\n\n"
        else:
            report_title = ""
        
        # Add to conversation
        AppSessionState.add_assistant_message(f"{report_title}{quality_report}")
        st.rerun()
    
    def handle_duplicate_detection(self, df: pd.DataFrame, first_name_col: str, last_name_col: str, threshold: str = 'medium'):
        """
        Handle duplicate record detection request.
        
        Args:
            df (DataFrame): DataFrame to check for duplicates
            first_name_col (str): Column name for first names
            last_name_col (str): Column name for last names
            threshold (str): Similarity threshold level
        """
        # Import the duplicate detector
        from duplicate_detector import DuplicateDetector
        
        # Create detector instance
        detector = DuplicateDetector()
        
        # Perform duplicate detection
        with st.spinner("Detecting duplicate records..."):
            duplicate_df, summary = detector.detect_name_duplicates(
                df, 
                first_name_col, 
                last_name_col, 
                threshold=threshold
            )
            
            # Generate report
            report = detector.generate_duplicate_report(duplicate_df, summary)
            
            # Store the results
            st.session_state.duplicate_results = duplicate_df
            st.session_state.duplicate_summary = summary
            
            # Add to conversation
            AppSessionState.add_assistant_message(
                f"I've analyzed the data for duplicate records based on name similarity.\n\n{report}"
            )
            
            # If we have duplicates, show them in a data preview
            if not duplicate_df.empty:
                st.session_state.show_duplicate_preview = True
            
            st.rerun()


    def handle_pdf_question(self, question: str):
        """
        Handle question about PDF content.
        
        Args:
            question: Question to answer
        """
        # Add user question to conversation
        AppSessionState.add_user_message(question)
        
        with st.spinner("Searching the PDF for answers..."):
            # Answer the question using the RAG system
            result = st.session_state.rag_system.answer_question(question)
            
            # Format the answer with citations
            if False:
                answer = f"{result['answer']}\n\n**Sources:**\n"
                for i, source in enumerate(result["sources"], 1):
                    answer += f"{i}. {source}\n"
            else:
                answer = result["answer"]
            
            # Add the answer to conversation
            AppSessionState.add_assistant_message(answer)
        
        st.rerun()
    
    def process_user_message(self, user_input: str):
        """
        Process user message and generate appropriate response.
        
        Args:
            user_input: User's message text
        """
        if not user_input.strip():
            st.warning("Please enter a message")
            return
        
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
            return
        
        # Add user message to conversation
        AppSessionState.add_user_message(user_input)
        
        # Check if user is asking to clear chat
        if MessageHandler.is_asking_to_clear_chat(user_input):
            # Clear the conversation history but save the user's clear request
            st.session_state.conversation = [{"role": "user", "content": user_input}]
            
            # Add AI response confirming the chat has been cleared
            AppSessionState.add_assistant_message("Chat history has been cleared!")
            st.rerun()
            return
        
        # Check if user is asking for data quality check
        elif MessageHandler.is_asking_for_data_quality_check(user_input):
            # Check if we have any files loaded
            if st.session_state.file_content is not None:
                # Generate data quality report for active file
                self.handle_data_quality_check(st.session_state.file_content)
                return
            elif st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                # Generate data quality report for the active file
                file_info = st.session_state.uploaded_files[st.session_state.active_file_id]
                self.handle_data_quality_check(file_info['content'], file_info['name'])
                return
            else:
                # No file loaded
                AppSessionState.add_assistant_message("No file is currently loaded. Please load a file first before requesting a data quality check.")
                st.rerun()
                return
        
        # Check if user is asking for duplicate detection
        elif MessageHandler.is_asking_for_duplicate_detection(user_input):
            # Show duplicate detection settings UI
            st.session_state.show_duplicate_settings = True
            
            # Check if we have any files loaded
            if st.session_state.file_content is not None or (
                st.session_state.active_file_id and 
                st.session_state.active_file_id in st.session_state.uploaded_files
            ):
                # Add AI response
                AppSessionState.add_assistant_message(
                    "I can help you detect duplicate records based on name similarity. "
                    "Please select the name columns and detection settings below."
                )
            else:
                # No file loaded
                AppSessionState.add_assistant_message(
                    "No file is currently loaded. Please load a file first before "
                    "requesting duplicate record detection."
                )
            
            st.rerun()
            return

        # Check if user is asking for DataFrame operations
        elif MessageHandler.is_asking_for_dataframe_operations(user_input):
            # Show DataFrame operations UI
            st.session_state.show_df_operations = True
            
            # Count available files
            available_files = len(st.session_state.uploaded_files)
            
            if available_files > 0:
                # Add AI response
                AppSessionState.add_assistant_message("I can help you perform DataFrame operations like merge, join, concat, fillna, and more. Please select the operation and files from the DataFrame Operations panel below.")
            else:
                # No files available
                AppSessionState.add_assistant_message("No files are currently loaded. Please upload files first before performing DataFrame operations.")
            
            st.rerun()
            return
        
        # Handle PDF URL requests or preferences
        elif "provide a direct pdf url" in user_input.lower() or "provide pdf url" in user_input.lower():
            st.session_state.show_pdf_url_input = True
            st.session_state.pdf_download_purpose = "general"
            
            # Add AI response
            AppSessionState.add_assistant_message("Please enter the URL of the PDF file you'd like to download.")
            st.rerun()
            return
        
        # Check if user is asking for direct PDF download
        elif MessageHandler.is_asking_for_pdf_download(user_input):
            st.session_state.show_pdf_url_input = True
            st.session_state.pdf_download_purpose = "general"
            
            # Add AI response
            AppSessionState.add_assistant_message("I'd be happy to help you download a PDF. Please enter the URL of the PDF file you'd like to download.")
            st.rerun()
            return
        
        # Check if user is asking for marketing data
        elif MessageHandler.is_asking_for_marketing_data(user_input):
            st.session_state.show_pdf_url_input = True
            st.session_state.pdf_download_purpose = "marketing"
            
            # Add AI response
            AppSessionState.add_assistant_message("I'd be happy to help with marketing data. Please provide the URL of the marketing report or document you'd like me to download for analysis.")
            st.rerun()
            return
        
        # Check if user is asking for company comparison
        elif MessageHandler.is_asking_for_comparison(user_input):
            company_name = PDFRequestParser.extract_company_name(user_input)
            
            if company_name:
                # Offer both options - auto-find or direct URL input
                AppSessionState.add_assistant_message(f"I'll help you compare with {company_name}'s annual report. Would you like me to automatically find the report or would you prefer to provide a direct PDF URL?")
            else:
                # If no company name could be extracted, offer direct URL option
                AppSessionState.add_assistant_message("I'd be happy to help with the comparison. Would you like to provide a company name for me to find the report, or do you already have a PDF URL?")
            
            st.rerun()
            return
        
        # Check if user is asking to load a file
        elif "load" in user_input.lower() and any(word in user_input.lower() for word in ["csv", "excel", "xls", "xlsx", "parquet", "file", "data"]):
            # Show file input
            st.session_state.show_file_input = True
            
            # Add AI response
            AppSessionState.add_assistant_message("Please upload the file(s) you want to analyze. You can select multiple files by holding down the Ctrl (or Cmd) key while selecting.")
            st.rerun()
            return
        
        # Check if user is asking to save/export a file
        elif any(word in user_input.lower() for word in ["save", "export", "write", "output"]) and any(word in user_input.lower() for word in ["csv", "excel", "xls", "xlsx", "parquet", "file", "data"]):
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
                if any(word in user_input.lower() for word in ["filter", "where", "condition"]):
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
                    USER REQUEST: {user_input}
                    
                    FILE INFO:
                    {file_info}
                    
                    Generate Python code to filter the data as requested. The filtered DataFrame should be assigned to 'filtered_df'.
                    """
                    
                    response = self.openai_handler.get_response(prompt, system_message)
                    
                    # Extract code from the response
                    code_block = MessageHandler.extract_code(response)
                    
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
                AppSessionState.add_assistant_message("Please enter the path where you want to export the file and select the format.")
                st.rerun()
                return
            else:
                # No file loaded
                AppSessionState.add_assistant_message("No file is currently loaded. Please load a file first before trying to export.")
                st.rerun()
                return
        
        # Check if the user is asking a question about the processed PDF
        elif MessageHandler.is_asking_about_pdf(user_input, st.session_state.pdf_processed):
            # Show a spinner while processing
            with st.spinner("Searching the PDF for answers..."):
                # Answer the question using the RAG system
                result = st.session_state.rag_system.answer_question(user_input)
                
                # Format the answer with citations
                if False:
                    answer = f"{result['answer']}\n\n**Sources:**\n"
                    for i, source in enumerate(result["sources"], 1):
                        answer += f"{i}. {source}\n"
                else:
                    answer = result["answer"]
                
                # Add the answer to conversation
                AppSessionState.add_assistant_message(answer)
            
            st.rerun()
            return
        
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
            response = self.openai_handler.get_response(
                prompt=f"""
                USER REQUEST: {user_input + annual_report_context}
                
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
            AppSessionState.add_assistant_message(response)
            
            # Extract code from the response
            code_block = MessageHandler.extract_code(response)
            
            if code_block:
                # Execute the code silently
                result = CodeExecutor.execute_code(code_block, df)
                st.session_state.code_execution_result = result
                
                # If there were errors, add them to the response
                if result['errors'] and not result['success']:
                    error_message = f"⚠️ There was an error executing the analysis: {result['errors']}"
                    AppSessionState.add_assistant_message(error_message)
            
            st.rerun()
            return
        else:
            # For general questions, get response from OpenAI
            prompt = f"User: {user_input}\nProvide a helpful response. If the user is asking about data analysis, suggest they load a CSV file first."
            
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
            
            response = self.openai_handler.get_response(prompt)
            
            # Add AI response to conversation
            AppSessionState.add_assistant_message(response)
            st.rerun()
            return
    
    def run(self):
        """Run the application."""
        # Create a two-column layout
        col1, col2 = st.columns([3, 2])
        
        # Render sidebar
        AppUIManager.render_sidebar(self.openai_handler)
        
        # Left column (Chat interface)
        with col1:
            # Render chat interface with conversation history
            AppUIManager.render_chat_interface()
            
            # Handle file upload
            uploaded_files = AppUIManager.render_file_uploader()
            if uploaded_files:
                self.handle_file_upload(uploaded_files)
            
            # Handle PDF URL input
            pdf_url, custom_filename, download_button = AppUIManager.render_pdf_url_input()
            if download_button and pdf_url:
                self.handle_pdf_download(pdf_url, custom_filename)
            
            # Handle company input
            company_name, find_report_button, provide_url_button = AppUIManager.render_company_input()
            if find_report_button and company_name:
                self.handle_company_report_request(company_name)
            elif provide_url_button:
                st.session_state.show_company_input = False
                st.session_state.show_pdf_url_input = True
                st.session_state.pdf_download_purpose = "comparison"
                if company_name:
                    st.session_state.comparison_company = company_name
                AppSessionState.add_assistant_message(
                    f"Please provide the direct URL to {company_name if company_name else 'the company'}'s annual report PDF."
                )
                st.rerun()
            
            # Handle DataFrame operations
            selected_file_ids, operation_type, params, execute_button = AppUIManager.render_dataframe_operations()
            if execute_button and selected_file_ids and operation_type and params:
                self.handle_dataframe_operation(selected_file_ids, operation_type, params)
            
            # Handle file export
            export_path, export_format, export_button = AppUIManager.render_export_form()
            if export_button and export_path and export_format:
                self.handle_export_request(export_path, export_format)
            

            # Handle duplicate detection settings
            first_name_col, last_name_col, threshold, detect_button = AppUIManager.render_duplicate_settings()
            if detect_button and first_name_col and last_name_col:
                if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                    df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
                    self.handle_duplicate_detection(df, first_name_col, last_name_col, threshold)
                elif st.session_state.file_content is not None:
                    self.handle_duplicate_detection(st.session_state.file_content, first_name_col, last_name_col, threshold)


            # Handle message form
            user_input, submit_button = AppUIManager.render_message_form()
            if submit_button:
                self.process_user_message(user_input)
            
            # Handle comparison follow-up buttons if the latest message is about comparison
            if st.session_state.conversation and len(st.session_state.conversation) >= 2:
                last_user_msg = ""
                for msg in reversed(st.session_state.conversation):
                    if msg["role"] == "user":
                        last_user_msg = msg["content"]
                        break
                
                if MessageHandler.is_asking_for_comparison(last_user_msg):
                    company_name = PDFRequestParser.extract_company_name(last_user_msg)
                    
                    # Add buttons for the two options
                    col1a, col1b = st.columns(2)
                    with col1a:
                        if st.button("Find automatically", key="auto_find"):
                            st.session_state.show_company_input = True
                            AppSessionState.add_assistant_message("Please confirm the company name or enter a different one.")
                            st.rerun()
                    with col1b:
                        if st.button("Provide PDF URL", key="provide_url"):
                            st.session_state.show_pdf_url_input = True
                            st.session_state.pdf_download_purpose = "comparison"
                            if company_name:
                                st.session_state.comparison_company = company_name
                            AppSessionState.add_assistant_message(f"Please provide the direct URL to {company_name if company_name else 'the company'}'s annual report PDF.")
                            st.rerun()
        
        # Right column (Data & Analysis)
        with col2:
            # Render data preview
            check_quality, show_operations = AppUIManager.render_data_preview()
            
            if check_quality:
                if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                    file_info = st.session_state.uploaded_files[st.session_state.active_file_id]
                    self.handle_data_quality_check(file_info['content'], file_info['name'])
                else:
                    self.handle_data_quality_check(st.session_state.file_content)
            
            if show_operations:
                st.session_state.show_df_operations = True
                AppSessionState.add_assistant_message("Please select the DataFrame operation you'd like to perform using the panel below.")
                st.rerun()
            
            # Render annual report info
            pdf_question = AppUIManager.render_annual_report_info()
            if isinstance(pdf_question, str) and pdf_question:
                self.handle_pdf_question(pdf_question)
            elif pdf_question is True:  # Process PDF button was clicked
                with st.spinner("Processing PDF for question answering..."):
                    success, message = st.session_state.rag_system.process_pdf(st.session_state.annual_report_path)
                    st.session_state.pdf_processed = success
                    if success:
                        st.success("PDF processed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to process PDF: {message}")
            
            # Render code execution results
            # Render duplicate detection results if available
            if st.session_state.show_duplicate_preview:
                AppUIManager.render_duplicate_preview()
            AppUIManager.render_code_execution_results()