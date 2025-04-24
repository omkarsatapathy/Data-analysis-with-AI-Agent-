import streamlit as st
import datetime
import os

class AppSessionState:
    """
    Manages the application's session state.
    
    This class encapsulates the initialization and management of the
    application's session state variables, making it easier to track 
    conversation history, loaded files, active operations, and other
    stateful aspects of the application.
    """
    @staticmethod
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
        
        # Generated files support
        if "generated_files" not in st.session_state:
            st.session_state.generated_files = {}  # dict to store generated files {id: {name, path, type}}
        
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
        
        # New duplicate detection workflow states
        if "column_transform_triggered_by_duplicate" not in st.session_state:
            st.session_state.column_transform_triggered_by_duplicate = False
        if "duplicate_first_col" not in st.session_state:
            st.session_state.duplicate_first_col = None
        if "duplicate_second_col" not in st.session_state:
            st.session_state.duplicate_second_col = None
        if "duplicate_threshold" not in st.session_state:
            st.session_state.duplicate_threshold = 'medium'
        if "duplicate_convert_to_string" not in st.session_state:
            st.session_state.duplicate_convert_to_string = True
        if "column_transform_attempted" not in st.session_state:
            st.session_state.column_transform_attempted = False

        # UI control states
        if "show_duplicate_column_mapping" not in st.session_state:
            st.session_state.show_duplicate_column_mapping = False
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

        if "continue_duplicate_detection" not in st.session_state:
            st.session_state.continue_duplicate_detection = False
        if "combined_df_for_duplicates" not in st.session_state:
            st.session_state.combined_df_for_duplicates = None
        
        # Column transformation states
        if "show_column_transform_input" not in st.session_state:
            st.session_state.show_column_transform_input = False
        if "transformed_df" not in st.session_state:
            st.session_state.transformed_df = None
        if "transformation_report" not in st.session_state:
            st.session_state.transformation_report = None
        if "show_transformation_preview" not in st.session_state:
            st.session_state.show_transformation_preview = False
        
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
        st.session_state.show_column_transform_input = False