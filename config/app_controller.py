import streamlit as st
import pandas as pd
import uuid
import os
import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable

# Import project modules
from config.app_session_state import AppSessionState
from config.app_ui_manager import AppUIManager
from config.app_file_manager import AppFileManager
from config.app_data_operations import AppDataOperations
from config.app_column_transformer import AppColumnTransformer
from config.app_pdf_manager import AppPDFManager
from config.app_duplicate_detector import AppDuplicateDetector
from config.app_code_execution import AppCodeExecution

# Import src modules
from src.llm import MessageHandler, OpenAIHandler
from src.duplicate_detector import DuplicateDetector
from src.data_operation import DataFrameOperations, DataAnalyzer
from src.pdf_fetcher import PDFDownloader, AnnualReportAgent, PDFRequestParser
from src.rag_app import RAGSystem, DocumentQASession
from src.file_export import FileHandler, FileExporter
from src.python_engine import CodeExecutor
from src.column_transformer import ColumnTransformer
# Publish the controller 24 April, 12:26PM 
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
    
    def handle_column_transformation(self, mapping_file, export_after_transform: bool = False):
        """
        Handle column transformation based on a mapping file.
        
        Args:
            mapping_file: Uploaded mapping file with old and new column names
            export_after_transform: Whether to show export dialog after transformation
        """
        if not mapping_file:
            st.error("Please upload a mapping file")
            return False
            
        try:
            # Check if this transformation was triggered by duplicate detection
            triggered_by_duplicate_detection = st.session_state.get('column_transform_triggered_by_duplicate', False)
            
            # Get the active DataFrame
            df = None
            if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
                file_name = st.session_state.uploaded_files[st.session_state.active_file_id]['name']
            elif st.session_state.file_content is not None:
                df = st.session_state.file_content
                file_name = st.session_state.file_path
            
            if df is None:
                st.error("No data loaded for transformation")
                return False
            
            # Load the mapping file
            mapping_df, message = ColumnTransformer.load_mapping_file(mapping_file)
            
            if mapping_df is None:
                st.error(message)
                return False
            
            # Validate the mapping
            validation_results = ColumnTransformer.validate_mapping(mapping_df, df)
            
            # Show warnings if any
            for warning in validation_results.get("warnings", []):
                st.warning(f"⚠️ {warning}")
            
            # Show errors if any and stop if validation failed
            if not validation_results.get("valid", True):
                for error in validation_results.get("errors", []):
                    st.error(f"❌ {error}")
                return False
            
            # Apply the transformation
            transformed_df, report = ColumnTransformer.transform_columns(df, mapping_df)
            
            # Generate a readable report
            report_text = ColumnTransformer.generate_transformation_report(report)
            
            # Add success message to conversation
            if report.get("transformed_count", 0) > 0:
                AppSessionState.add_assistant_message(
                    f"Successfully transformed {report['transformed_count']} column names in '{file_name}'."
                )
            else:
                AppSessionState.add_assistant_message(
                    f"No columns were transformed. This could be because none of the column names in the mapping file "
                    f"matched the columns in '{file_name}'."
                )
            
            # Handle differently based on whether this was triggered by duplicate detection
            if triggered_by_duplicate_detection:
                # Update the current file with the transformed dataframe
                if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                    st.session_state.uploaded_files[st.session_state.active_file_id]['content'] = transformed_df
                else:
                    st.session_state.file_content = transformed_df
                
                # Reset the duplicate detection trigger flag
                st.session_state.column_transform_triggered_by_duplicate = False
                
                # Get parameters for duplicate detection
                first_col = st.session_state.get('duplicate_first_col')
                second_col = st.session_state.get('duplicate_second_col')
                threshold = st.session_state.get('duplicate_threshold', 'medium')
                convert_to_string = st.session_state.get('duplicate_convert_to_string', True)
                
                # Get all DataFrames for combination
                dfs = []
                for file_id, file_info in st.session_state.uploaded_files.items():
                    dfs.append(file_info['content'])
                
                # Combine into a single DataFrame
                combined_df = pd.concat(dfs, axis=0, ignore_index=True) if len(dfs) > 1 else dfs[0]
                
                # Hide the column transformation input panel
                st.session_state.show_column_transform_input = False
                
                # Proceed directly with duplicate detection
                self.handle_duplicate_detection(
                    combined_df,
                    first_col,
                    second_col,
                    threshold,
                    convert_to_string
                )
            else:
                # Normal transformation flow - show preview and export option
                st.session_state.transformed_df = transformed_df
                st.session_state.transformation_report = report_text
                st.session_state.show_transformation_preview = True
                
                # Option to export after transformation
                if export_after_transform:
                    st.session_state.filtered_df = transformed_df
                    st.session_state.show_export_input = True
                
                # Hide the transformation input panel
                st.session_state.show_column_transform_input = False
            
            return True
            
        except Exception as e:
            import traceback
            st.error(f"Error transforming columns: {str(e)}")
            st.text_area("Error details", value=traceback.format_exc(), height=150)
            AppSessionState.add_assistant_message(
                f"An error occurred while applying column transformations: {str(e)}. Please check your mapping file and try again."
            )
            return False
        
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
        df_to_export = None
        
        # Try all possible sources of the DataFrame in priority order
        if st.session_state.transformed_df is not None:
            # First priority: transformed DataFrame
            df_to_export = st.session_state.transformed_df
            context = "transformed"
        elif st.session_state.filtered_df is not None:
            # Second priority: filtered DataFrame
            df_to_export = st.session_state.filtered_df
            context = "filtered"
        elif st.session_state.operation_result is not None:
            # Third priority: operation result
            df_to_export = st.session_state.operation_result
            context = "operation result"
        elif st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
            # Fourth priority: active file
            df_to_export = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
            context = "active file"
        elif st.session_state.file_content is not None:
            # Fallback: file content
            df_to_export = st.session_state.file_content
            context = "file"
        
        if df_to_export is not None and isinstance(df_to_export, pd.DataFrame):
            # Export the DataFrame using the mapped format
            message, success = FileExporter.export_dataframe(
                df_to_export, 
                export_path, 
                export_format
            )
            
            if success:
                # Add success message to conversation
                AppSessionState.add_assistant_message(
                    f"{message} I exported the {context} with {df_to_export.shape[0]} rows and {df_to_export.shape[1]} columns."
                )
                
                # Add to generated files
                file_id = str(uuid.uuid4())
                st.session_state.generated_files[file_id] = {
                    'name': os.path.basename(export_path),
                    'path': export_path,
                    'type': export_format
                }
                
                # Hide export input
                st.session_state.show_export_input = False
                st.session_state.filtered_df = None
                st.session_state.transformed_df = None  # Reset transformed DataFrame after export
                st.session_state.show_transformation_preview = False
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
    
    def handle_duplicate_detection(self, df: pd.DataFrame, first_col: str, second_col: str, threshold: str = 'medium', convert_to_string: bool = True):
        """
        Handle duplicate record detection request.
        
        Args:
            df (DataFrame): DataFrame to check for duplicates
            first_col (str): First column name for comparison
            second_col (str): Second column name for comparison
            threshold (str): Similarity threshold level
            convert_to_string (bool): Whether to convert columns to string
        """
        # Check if columns exist in the dataframe
        if first_col not in df.columns or second_col not in df.columns:
            # Check if we already attempted transformation (to avoid infinite loops)
            if st.session_state.get('column_transform_attempted', False):
                # Reset the flag
                st.session_state.column_transform_attempted = False
                
                # Show error message
                st.error(f"Columns '{first_col}' and/or '{second_col}' still not found after transformation.")
                AppSessionState.add_assistant_message(
                    f"I couldn't find the columns '{first_col}' and/or '{second_col}' even after column transformation. "
                    f"Please ensure these columns exist in your data or choose different columns for duplicate detection."
                )
                return
        
        # Check if we have multiple files and need to handle column mismatches
        multiple_files = len(st.session_state.uploaded_files) > 1
        
        if multiple_files:
            # Check if columns exist in all files
            all_columns_exist = True
            missing_columns = {}
            
            for file_id, file_info in st.session_state.uploaded_files.items():
                file_df = file_info['content']
                if first_col not in file_df.columns or second_col not in file_df.columns:
                    all_columns_exist = False
                    missing_columns[file_id] = {
                        'name': file_info['name'],
                        'columns': [col for col in [first_col, second_col] if col not in file_df.columns]
                    }
            
            # If we have missing columns, trigger column transformation
            if not all_columns_exist:
                # Set the transformation attempted flag to prevent infinite loops
                st.session_state.column_transform_attempted = True
                
                # Store duplicate detection parameters for later use
                st.session_state.duplicate_first_col = first_col
                st.session_state.duplicate_second_col = second_col
                st.session_state.duplicate_threshold = threshold
                st.session_state.duplicate_convert_to_string = convert_to_string
                
                # Set flag to indicate transformation triggered by duplicate detection
                st.session_state.column_transform_triggered_by_duplicate = True
                
                # Show column transformation interface
                st.session_state.show_column_transform_input = True
                
                # Add message about column mismatch
                missing_files = [f"'{info['name']}' (missing: {', '.join(info['columns'])})" 
                                for info in missing_columns.values()]
                
                AppSessionState.add_assistant_message(
                    f"I've detected column name mismatches across your files. "
                    f"Before we can detect duplicates, we need to standardize column names.\n\n"
                    f"The columns '{first_col}' and '{second_col}' are not present in all files. "
                    f"Missing in: {', '.join(missing_files)}.\n\n"
                    f"Please upload a mapping file or use the template to map column names."
                )
                
                return
        
        # Reset the transformation attempted flag
        st.session_state.column_transform_attempted = False
        
        # Create detector instance
        detector = DuplicateDetector()
        
        # Perform duplicate detection
        with st.spinner("Detecting duplicate records..."):
            # Make a copy of the DataFrame to avoid modifying the original
            df_copy = df.copy()
            
            # Convert columns to string if specified
            if convert_to_string:
                df_copy[first_col] = df_copy[first_col].astype(str)
                df_copy[second_col] = df_copy[second_col].astype(str)
                
            # Perform duplicate detection
            duplicate_df, summary = detector.detect_name_duplicates(
                df_copy, 
                first_col,  # Using as first_name_col
                second_col, # Using as last_name_col 
                threshold=threshold
            )
            
            # Generate report
            report = detector.generate_duplicate_report(duplicate_df, summary)
            
            # Store the results
            st.session_state.duplicate_results = duplicate_df
            st.session_state.duplicate_summary = summary
            
            # Add to conversation
            if not duplicate_df.empty:
                AppSessionState.add_assistant_message(
                    f"I've analyzed the data for duplicate records based on columns '{first_col}' and '{second_col}'.\n\n"
                    f"Found {summary['total_duplicates']} potential duplicate records in {summary['duplicate_groups']} groups.\n\n"
                    f"You can view the summary and download the results from the panel on the right."
                )
                
                # Show preview with download button
                st.session_state.show_duplicate_preview = True
            else:
                AppSessionState.add_assistant_message(
                    f"I've analyzed the data for duplicate records based on columns '{first_col}' and '{second_col}', but no duplicates were found."
                )
            
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
                    "I can help you detect duplicate records based on column similarity. "
                    "Please select the columns and detection settings below."
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
            
        # Check if user is asking for column transformation
        elif MessageHandler.is_asking_for_column_transformation(user_input):
            # Show column transformation UI
            st.session_state.show_column_transform_input = True
            
            # Check if we have any files loaded
            if st.session_state.file_content is not None or (
                st.session_state.active_file_id and 
                st.session_state.active_file_id in st.session_state.uploaded_files
            ):
                # Add AI response
                AppSessionState.add_assistant_message(
                    "I can help you transform column names. Please upload a mapping file with two columns: "
                    "'old_column_name' and 'new_column_name'. You can download a template from the panel below."
                )
            else:
                # No file loaded
                AppSessionState.add_assistant_message(
                    "No file is currently loaded. Please load a file first before "
                    "requesting column transformation."
                )
            
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
            if st.session_state.transformed_df is not None:
                # First priority: transformed DataFrame
                df_to_export = st.session_state.transformed_df
            elif st.session_state.operation_result is not None:
                # Second priority: operation result
                df_to_export = st.session_state.operation_result
            elif st.session_state.file_content is not None:
                # Third priority: traditional single file content
                df_to_export = st.session_state.file_content
            elif st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                # Fourth priority: active file from multiple files
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
                
                # Format the answer
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
            
            # Include info about generated files
            if st.session_state.generated_files:
                prompt += "\n\nGenerated files:"
                for file_id, file_info in st.session_state.generated_files.items():
                    prompt += f"\n- {file_info['name']} ({file_info['type']})"
            
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
            uploaded_files = AppFileManager.render_file_uploader()
            if uploaded_files:
                self.handle_file_upload(uploaded_files)
            
            # Handle column transformation
            mapping_file, transform_button, export_after_transform = AppColumnTransformer.render_column_transform_input()
            if transform_button and mapping_file:
                self.handle_column_transformation(mapping_file, export_after_transform)
            
            # Handle PDF URL input
            pdf_url, custom_filename, download_button = AppPDFManager.render_pdf_url_input()
            if download_button and pdf_url:
                self.handle_pdf_download(pdf_url, custom_filename)
            
            # Handle company input
            company_name, find_report_button, provide_url_button = AppPDFManager.render_company_input()
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
            selected_file_ids, operation_type, params, execute_button = AppDataOperations.render_dataframe_operations()
            if execute_button and selected_file_ids and operation_type and params:
                self.handle_dataframe_operation(selected_file_ids, operation_type, params)
            
            # Handle file export
            export_path, export_format, export_button = AppFileManager.render_export_form()
            if export_button and export_path and export_format:
                self.handle_export_request(export_path, export_format)
            
            # Handle duplicate detection settings
            first_col, second_col, threshold, convert_to_string, detect_button, selected_file_ids = AppDuplicateDetector.render_duplicate_settings()

            if detect_button and first_col and second_col:
                if selected_file_ids and len(selected_file_ids) > 0:
                    dfs = []
                    for file_id in selected_file_ids:
                        if file_id in st.session_state.uploaded_files:
                            dfs.append(st.session_state.uploaded_files[file_id]['content'])
                    
                    if dfs:
                        if len(dfs) > 1:
                            combined_df = pd.concat(dfs, axis=0, ignore_index=True)
                        else:
                            combined_df = dfs[0]
                        
                        self.handle_duplicate_detection(combined_df, first_col, second_col, threshold, convert_to_string)
                elif st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                    df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
                    self.handle_duplicate_detection(df, first_col, second_col, threshold, convert_to_string)
                elif st.session_state.file_content is not None:
                    self.handle_duplicate_detection(st.session_state.file_content, first_col, second_col, threshold, convert_to_string)
            
            # Handle export transformed data
            export_path, export_format, export_button = AppFileManager.render_transformation_preview()
            if export_button and export_path:
                self.handle_export_request(export_path, export_format)
            
            # Check if we need to continue with duplicate detection after column transformation
            if st.session_state.get('continue_duplicate_detection', False) and st.session_state.get('combined_df_for_duplicates') is not None:
                # Get parameters for duplicate detection
                first_col = st.session_state.get('duplicate_first_col')
                second_col = st.session_state.get('duplicate_second_col')
                threshold = st.session_state.get('duplicate_threshold', 'medium')
                convert_to_string = st.session_state.get('duplicate_convert_to_string', True)
                
                # Reset flags
                st.session_state.continue_duplicate_detection = False
                combined_df = st.session_state.combined_df_for_duplicates
                st.session_state.combined_df_for_duplicates = None
                st.session_state.column_transform_triggered_by_duplicate = False
                
                # Perform duplicate detection on the combined dataframe
                self.handle_duplicate_detection(
                    combined_df,
                    first_col,
                    second_col,
                    threshold,
                    convert_to_string
                )
            
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
            check_quality, show_operations, show_transform = AppFileManager.render_data_preview()
            
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
                
            if show_transform:
                st.session_state.show_column_transform_input = True
                AppSessionState.add_assistant_message("Please upload a mapping file with old and new column names to transform your data.")
                st.rerun()
            
            # Render annual report info
            pdf_question = AppPDFManager.render_annual_report_info()
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
            
            # Render duplicate detection results if available
            if st.session_state.show_duplicate_preview:
                AppDuplicateDetector.render_duplicate_preview()
            
            # Render code execution results
            AppCodeExecution.render_code_execution_results()
        
    


__all__ = ['AppController']