import streamlit as st
import sys
import os

# Make sure the directory containing config is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.streamlit_config import load_streamlit_config

def main():
    """
    Main entry point for the Agentic AI Data Analyst application.
    
    This application combines AI-powered data analysis, visualization,
    document processing, and question answering in an interactive interface.
    Users can upload files, ask questions about their data, perform DataFrame
    operations, analyze PDF documents, and export results.
    """
    # Create and run application controller
    app = load_streamlit_config()
    app.run()

if __name__ == "__main__":
    main()