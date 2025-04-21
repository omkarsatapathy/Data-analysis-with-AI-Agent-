import streamlit as st
import pandas as pd
import os
import datetime
import uuid
from typing import Dict, List, Tuple, Any, Optional, Callable

# Import from within the project
from src.llm import OpenAIHandler

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
        10. Transform column names using mapping files
        """)
    
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
            
            # Generated files section
            if st.session_state.generated_files:
                st.subheader("Generated Files")
                # Display all generated files
                for file_id, file_info in st.session_state.generated_files.items():
                    st.markdown(f"- {file_info['name']} ({file_info['type']})")
                
                if st.button("Clear all generated files", key="clear_all_generated_files"):
                    st.session_state.generated_files = {}
                    st.rerun()
            
            # Annual report information
            if st.session_state.annual_report_path:
                st.success(f"File loaded: {os.path.basename(st.session_state.annual_report_path)}")
                if st.button("Clear report", key="clear_report"):
                    st.session_state.annual_report_path = None
                    st.session_state.comparison_company = None
                    st.rerun()
            
            # Clear chat button in sidebar
            if st.button("Clear Chat History", key="clear_chat_button"):
                from config.app_session_state import AppSessionState
                AppSessionState.reset_conversation()
                st.success("Chat history has been cleared.")
                st.rerun()
            
            # About section
            st.divider()
            AppUIManager.render_about_section()
    
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
                    from src.llm import MessageHandler
                    content = MessageHandler.clean_response(content)
                st.write(f"**AI**: {content}")
    
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