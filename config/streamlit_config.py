import streamlit as st
import sys
import os

# Make sure the parent directory is in the path so we can use absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_session_state import AppSessionState
from config.app_controller import AppController

# Main entry point for streamlit configuration
def load_streamlit_config():
    """Initialize the Streamlit app configuration."""
    app = AppController()
    return app