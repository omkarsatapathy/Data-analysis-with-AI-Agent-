import streamlit as st
import pandas as pd
import os
from typing import Tuple, Any, List

class AppFileManager:
    """
    Manages file-related UI components and interactions.
    
    This class encapsulates file uploading, previewing, and exporting
    functionality in the UI.
    """
    
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
                
                # Add column transformation button
                show_transform = st.button("Transform Column Names", key="transform_columns")
            
            return check_quality, show_operations, show_transform
        
        # Display operation result if available
        elif st.session_state.operation_result is not None:
            st.subheader("Operation Result")
            st.dataframe(st.session_state.operation_result.head(10))
            
            # Display operation details
            with st.expander("Operation Details"):
                st.markdown(st.session_state.operation_description)
            
            return False, False, False
        
        # Display transformed DataFrame if available
        elif st.session_state.transformed_df is not None:
            st.subheader("Column Transformation Result")
            st.dataframe(st.session_state.transformed_df.head(10))
            
            # Display transformation details
            with st.expander("Transformation Details"):
                if st.session_state.transformation_report:
                    st.markdown(st.session_state.transformation_report)
            
            # Add option to export the transformed data
            export_button = st.button("Export Transformed Data", key="export_transformed_data")
            if export_button:
                st.session_state.filtered_df = st.session_state.transformed_df
                st.session_state.show_export_input = True
                st.rerun()
            
            return False, False, False
        
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
                
                # Add column transformation button
                show_transform = st.button("Transform Column Names", key="transform_columns")
            
            return check_quality, False, show_transform
        
        return False, False, False
    
    @staticmethod
    def render_transformation_preview():
        """Render the column transformation results preview."""
        if st.session_state.show_transformation_preview and st.session_state.transformed_df is not None:
            # Check if triggered by duplicate detection
            triggered_by_duplicate = st.session_state.get('column_transform_triggered_by_duplicate', False)
            
            title = "Column Transformation" if not triggered_by_duplicate else "Column Standardization for Duplicate Detection"
            st.subheader(title)
            
            # Show a before/after comparison
            col1, col2 = st.columns(2)
            
            # Get the original DataFrame
            original_df = None
            if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                original_df = st.session_state.uploaded_files[st.session_state.active_file_id]['content']
            elif st.session_state.file_content is not None:
                original_df = st.session_state.file_content
            
            with col1:
                st.write("### Original Columns")
                if original_df is not None:
                    st.write(", ".join(original_df.columns.tolist()))
            
            with col2:
                st.write("### New Columns")
                st.write(", ".join(st.session_state.transformed_df.columns.tolist()))
            
            # Show the report if available
            if st.session_state.transformation_report:
                st.markdown(st.session_state.transformation_report)
            
            # Only show export options if not triggered by duplicate detection
            if not triggered_by_duplicate:
                # Export options
                st.write("### Export Transformed Data")
                export_format = st.selectbox(
                    "Select export format:",
                    options=["CSV", "XLSX (Excel)", "Parquet"],
                    index=0,
                    key="transform_export_format"
                )
                
                export_path = st.text_input("Export path (including filename):", key="transform_export_path")
                
                # Map the display format to actual format
                format_mapping = {
                    "CSV": "csv",
                    "XLSX (Excel)": "xlsx",
                    "Parquet": "parquet"
                }
                
                export_button = st.button("Export Transformed Data", key="export_transform_button")
            else:
                # If triggered by duplicate detection, show a different message and button
                st.info("""
                    Column names have been standardized for duplicate detection.
                    You can now proceed with duplicate detection using these standardized columns.
                """)
                
                # Continue button for duplicate detection
                export_button = False
                export_path = None
                export_format = None
                format_mapping = {}
                
                if st.button("Continue with Duplicate Detection", key="continue_duplicate_detection"):
                    # Clear transformation preview flag
                    st.session_state.show_transformation_preview = False
                    
                    # Update all files with transformed columns
                    if st.session_state.active_file_id and st.session_state.active_file_id in st.session_state.uploaded_files:
                        # Update the active file
                        st.session_state.uploaded_files[st.session_state.active_file_id]['content'] = st.session_state.transformed_df
                        
                        # Get params for duplicate detection
                        first_col = st.session_state.get('duplicate_first_col')
                        second_col = st.session_state.get('duplicate_second_col')
                        threshold = st.session_state.get('duplicate_threshold', 'medium')
                        convert_to_string = st.session_state.get('duplicate_convert_to_string', True)
                        
                        # Get all DataFrames for combination
                        dfs = []
                        for file_id, file_info in st.session_state.uploaded_files.items():
                            dfs.append(file_info['content'])
                        
                        # Combine all DataFrames
                        combined_df = pd.concat(dfs, axis=0, ignore_index=True) if len(dfs) > 1 else dfs[0]
                        
                        # Set trigger flag for app controller to continue with duplicate detection
                        st.session_state.continue_duplicate_detection = True
                        st.session_state.combined_df_for_duplicates = combined_df
                        st.rerun()
            
            # Option to close preview
            if st.button("Close Preview", key="close_transform_preview"):
                st.session_state.show_transformation_preview = False
                
                # Also clear duplicate detection flags if present
                if triggered_by_duplicate:
                    st.session_state.column_transform_triggered_by_duplicate = False
                
                st.rerun()
                    
            return export_path, format_mapping.get(export_format, None) if export_format else None, export_button
        
        return None, None, False