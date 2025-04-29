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
            # Check if this is triggered by duplicate detection
            triggered_by_duplicate = st.session_state.get('column_transform_triggered_by_duplicate', False)
            
            if triggered_by_duplicate:
                st.subheader("Column Standardization for Duplicate Detection")
                st.info("""
                    **Column name differences detected across files.** 
                    To detect duplicates, we need to standardize column names across all files.
                    Please upload a mapping file that maps the original column names to the target column names.
                """)
            else:
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
                
                # If triggered by duplicate detection, show all files and their columns
                if triggered_by_duplicate and len(st.session_state.uploaded_files) > 1:
                    with st.expander("All files columns (for reference)", expanded=True):
                        # Get target column names from duplicate detection
                        first_col = st.session_state.get('duplicate_first_col')
                        second_col = st.session_state.get('duplicate_second_col')
                        target_cols = [first_col, second_col]
                        
                        st.write("**Target columns for duplicate detection:**")
                        st.write(f"- First column: **{first_col}**")
                        st.write(f"- Second column: **{second_col}**")
                        st.write("---")
                        
                        for file_id, file_info in st.session_state.uploaded_files.items():
                            st.write(f"**{file_info['name']}**")
                            file_cols = file_info['content'].columns.tolist()
                            missing_cols = [col for col in target_cols if col not in file_cols]
                            
                            if missing_cols:
                                st.write(f"📌 Missing columns: {', '.join(['**'+col+'**' for col in missing_cols])}")
                            
                            st.write(f"All columns: {', '.join(file_cols)}")
                            st.write("---")
                
                # Upload mapping file
                st.write("### Upload Column Mapping File")
                mapping_description = """
                    Upload a CSV or Excel file with two columns: 'old_column_name' and 'new_column_name'.
                    Each row defines how a column should be renamed.
                """
                if triggered_by_duplicate:
                    mapping_description += """
                        \n- Make sure to include mappings for the duplicate detection columns.
                        - All files should have standardized column names after transformation.
                    """
                st.write(mapping_description)
                
                mapping_file = st.file_uploader(
                    "Choose column mapping file",
                    type=['csv', 'xlsx', 'xls'],
                    key="column_mapping_uploader"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Transform button
                    transform_button = st.button(
                        "Apply Column Transformation" if not triggered_by_duplicate else "Standardize Columns & Continue",
                        key="transform_columns_button"
                    )
                
                with col2:
                    # Export transformed data option - only if not triggered by duplicate detection
                    if not triggered_by_duplicate:
                        export_after_transform = st.checkbox(
                            "Export after transformation", 
                            value=True,
                            key="export_after_transform"
                        )
                    else:
                        export_after_transform = False
                
                # Template download button
                template_button_text = "Download Template" if not triggered_by_duplicate else "Download Mapping Template"
                if st.button(template_button_text, key="download_template_button"):
                    # Create sample template
                    if triggered_by_duplicate and len(st.session_state.uploaded_files) > 1:
                        # Get all unique columns across files
                        all_columns = set()
                        for file_id, file_info in st.session_state.uploaded_files.items():
                            all_columns.update(file_info['content'].columns)
                        
                        # Create template with target duplicate detection columns as defaults
                        template_df = pd.DataFrame(columns=['old_column_name', 'new_column_name'])
                        first_col = st.session_state.get('duplicate_first_col')
                        second_col = st.session_state.get('duplicate_second_col')
                        
                        # Add suggested mappings for all columns
                        for col in sorted(list(all_columns)):
                            template_df = pd.concat([template_df, pd.DataFrame({
                                'old_column_name': [col],
                                'new_column_name': [
                                    first_col if col.lower() == first_col.lower() or 'first' in col.lower() or 'fname' in col.lower() else 
                                    second_col if col.lower() == second_col.lower() or 'last' in col.lower() or 'lname' in col.lower() else 
                                    col
                                ]
                            })], ignore_index=True)
                    else:
                        # Regular template with current file columns
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
                
                # If triggered by duplicate detection, provide option to cancel
                if triggered_by_duplicate:
                    if st.button("Cancel Column Standardization", key="cancel_column_transform"):
                        st.session_state.show_column_transform_input = False
                        st.session_state.column_transform_triggered_by_duplicate = False
                        st.session_state.duplicate_first_col = None
                        st.session_state.duplicate_second_col = None
                        st.session_state.duplicate_threshold = 'medium'
                        st.session_state.duplicate_convert_to_string = True
                        st.rerun()
                
                return mapping_file, transform_button, export_after_transform
            else:
                st.warning("No data loaded. Please upload a file first.")
                # Return default values
                return None, False, False
        
        # Return default values when not showing column transform input
        return None, False, False