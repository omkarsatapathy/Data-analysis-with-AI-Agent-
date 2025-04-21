import streamlit as st
import pandas as pd
from typing import Tuple, Dict, Any, Optional

class AppColumnTransformer:
    """
    Manages the column transformation UI components.
    
    This class encapsulates UI elements for transforming column names
    based on mapping files.
    """
    
    @staticmethod
    def render_column_transform_input():
        """Render the column transformation input interface."""
        if st.session_state.show_column_transform_input:
            st.subheader("Column Transformation")
            
            # Check if we have any files loaded
            df = None
            if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
                active_file_name = st.session_state.uploaded_files[st.session_state.active_file_id]['name']
            elif st.session_state.file_content is not None:
                df = st.session_state.file_content
                active_file_name = st.session_state.file_path
            
            if df is not None:
                st.write(f"Active file: **{active_file_name}**")
                
                # Display current columns
                with st.expander("Current columns", expanded=True):
                    cols_df = pd.DataFrame({
                        'Column Name': df.columns,
                        'Data Type': [str(df[col].dtype) for col in df.columns]
                    })
                    st.dataframe(cols_df)
                
                # Upload mapping file
                st.write("### Upload Column Mapping File")
                st.write("Upload a CSV or Excel file with two columns: 'old_column_name' and 'new_column_name'")
                
                mapping_file = st.file_uploader(
                    "Choose column mapping file",
                    type=['csv', 'xlsx', 'xls'],
                    key="column_mapping_uploader"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Transform button
                    transform_button = st.button("Apply Column Transformation", key="transform_columns_button")
                
                with col2:
                    # Export transformed data option
                    export_after_transform = st.checkbox(
                        "Export after transformation", 
                        value=True,
                        key="export_after_transform"
                    )
                
                # Template download button
                if st.button("Download Template", key="download_template_button"):
                    # Create sample template
                    template_df = pd.DataFrame({
                        'old_column_name': df.columns[:3].tolist() + ['example_old_name'],
                        'new_column_name': ['new_' + col for col in df.columns[:3].tolist()] + ['example_new_name']
                    })
                    
                    # Convert to CSV
                    csv = template_df.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download Template CSV",
                        data=csv,
                        file_name="column_mapping_template.csv",
                        mime="text/csv",
                        key="download_template_csv"
                    )
                
                return mapping_file, transform_button, export_after_transform
            else:
                st.warning("No data loaded. Please upload a file first.")
        
        return None, False, False