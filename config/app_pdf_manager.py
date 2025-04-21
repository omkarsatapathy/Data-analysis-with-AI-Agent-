import streamlit as st
import os
from typing import Tuple, Optional

class AppPDFManager:
    """
    Manages the PDF-related UI components.
    
    This class encapsulates UI elements for PDF downloading, 
    company report fetching, and PDF Q&A functionality.
    """
    
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