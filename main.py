import streamlit as st
from streamlit_config import AppController

def main():
    """
    Main entry point for the Agentic AI Data Analyst application.
    
    This application combines AI-powered data analysis, visualization,
    document processing, and question answering in an interactive interface.
    Users can upload files, ask questions about their data, perform DataFrame
    operations, analyze PDF documents, and export results.
    """
    # Create and run application controller
    app = AppController()
    app.run()

if __name__ == "__main__":
    main()