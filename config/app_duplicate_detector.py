import streamlit as st
import pandas as pd
import uuid
import datetime
from typing import Tuple, Dict, Any, Optional

class AppDuplicateDetector:
    """
    Manages the duplicate detection UI components.
    
    This class encapsulates UI elements for detecting and displaying
    duplicate records in a dataset.
    """
    
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
                # Let user specify if they want to convert all columns to string
                convert_to_string = st.checkbox("Convert all columns to string before checking duplicates", value=True)
                
                # Let user select columns for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    first_col = st.selectbox(
                        "First Column:",
                        options=df.columns.tolist(),
                        key="first_col",
                        index=0
                    )
                    
                    # Show current data type
                    if first_col:
                        st.write(f"Current data type: {df[first_col].dtype}")
                
                with col2:
                    second_col = st.selectbox(
                        "Second Column:",
                        options=df.columns.tolist(),
                        key="second_col",
                        index=min(1, len(df.columns)-1) if len(df.columns) > 1 else 0
                    )
                    
                    # Show current data type
                    if second_col:
                        st.write(f"Current data type: {df[second_col].dtype}")
                
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
                
                return first_col, second_col, threshold, convert_to_string, detect_button
            else:
                st.warning("No data loaded. Please upload a file first.")
        
        return None, None, None, False, False
    
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
            
            # Show each group in an expander
            duplicate_df = st.session_state.duplicate_results
            for group_id in sorted(duplicate_df['duplicate_group'].unique()):
                group_df = duplicate_df[duplicate_df['duplicate_group'] == group_id]
                
                with st.expander(f"Group {int(group_id)} ({len(group_df)} records)"):
                    st.dataframe(group_df)
            
            # Add download button
            if st.button("Download Duplicate Records", key="download_duplicates"):
                # Generate a filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"/Users/omkar/Downloads/duplicate_records_{timestamp}.csv"
                
                # Save to CSV
                duplicate_df.to_csv(filename, index=False)
                
                # Add to generated files
                file_id = str(uuid.uuid4())
                st.session_state.generated_files[file_id] = {
                    'name': f"duplicate_records_{timestamp}.csv",
                    'path': filename,
                    'type': 'csv'
                }
                
                st.success(f"File saved to {filename}")
            
            # Option to close preview
            if st.button("Close Preview", key="close_duplicate_preview"):
                st.session_state.show_duplicate_preview = False
                st.rerun()