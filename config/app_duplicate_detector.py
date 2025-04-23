import streamlit as st
import pandas as pd
import uuid
import datetime
from typing import Tuple, Dict, Any, Optional, List

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
            
            if st.session_state.show_duplicate_column_mapping:
                return None, None, None, False, False, []
            
            # Check if any files are loaded
            if len(st.session_state.uploaded_files) > 0:
                st.write("### Select Files for Duplicate Detection")
                
                # Create a multiselect for files
                file_options = {file_id: file_info['name'] for file_id, file_info in st.session_state.uploaded_files.items()}
                
                selected_file_ids = st.multiselect(
                    "Select files to check for duplicates:",
                    options=list(file_options.keys()),
                    format_func=lambda x: file_options[x],
                    default=[st.session_state.active_file_id] if st.session_state.active_file_id else [],
                    key="duplicate_file_selector"
                )
                
                if not selected_file_ids:
                    st.warning("Please select at least one file for duplicate detection.")
                    return None, None, None, False, False, []
                
                # Get all column names from selected files
                all_columns = set()
                column_sets = []
                
                for file_id in selected_file_ids:
                    df = st.session_state.uploaded_files[file_id]['content']
                    column_set = set(df.columns.tolist())
                    column_sets.append((file_id, column_set))
                    all_columns.update(column_set)
                
                # Check if all files have the same columns
                have_same_columns = all(column_sets[0][1] == column_set[1] for column_set in column_sets[1:]) if len(column_sets) > 1 else True
                
                if not have_same_columns and len(selected_file_ids) > 1:
                    st.error("Selected files have different column names. Please use the Column Transformation feature to map them first.")
                    
                    # Show column differences
                    st.write("### Column Differences:")
                    for file_id, columns in column_sets:
                        st.write(f"**{file_options[file_id]}**: {', '.join(sorted(list(columns)))}")
                    
                    # Option to download mapping template
                    if st.button("Download Mapping Template", key="download_duplicate_mapping_template"):
                        # Create mapping template with columns from all files
                        all_columns_list = sorted(list(all_columns))
                        template_df = pd.DataFrame({
                            'old_column_name': all_columns_list,
                            'new_column_name': all_columns_list  # Default mapping is same name
                        })
                        
                        # Convert to CSV
                        csv = template_df.to_csv(index=False)
                        
                        # Create download button
                        st.download_button(
                            label="Download Template CSV",
                            data=csv,
                            file_name="column_mapping_template.csv",
                            mime="text/csv",
                            key="download_template_csv_duplicate"
                        )
                    
                    # Show transformation button
                    transform_button = st.button("Go to Column Transformation", key="go_to_column_transform")
                    if transform_button:
                        st.session_state.show_column_transform_input = True
                        st.session_state.show_duplicate_settings = False
                        st.rerun()
                    
                    return None, None, None, False, False, []
                
                # Get a reference DataFrame for column selection (use the first file)
                df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
                
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
                
                return first_col, second_col, threshold, convert_to_string, detect_button, selected_file_ids
            else:
                st.warning("No files loaded. Please upload files first.")
        
        return None, None, None, False, False, []
    
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
            
            # Show file breakdown if _source_file column exists
            duplicate_df = st.session_state.duplicate_results
            if "_source_file" in duplicate_df.columns:
                st.write("### Files containing duplicates:")
                file_counts = duplicate_df['_source_file'].value_counts().reset_index()
                file_counts.columns = ['File', 'Records']
                st.dataframe(file_counts)
            
            # Show each group in an expander
            for group_id in sorted(duplicate_df['duplicate_group'].unique()):
                group_df = duplicate_df[duplicate_df['duplicate_group'] == group_id]
                
                # Create a more descriptive header
                if "_source_file" in group_df.columns:
                    file_count = group_df['_source_file'].nunique()
                    files_str = "from multiple files" if file_count > 1 else "in same file"
                    header = f"Group {int(group_id)} ({len(group_df)} records {files_str})"
                else:
                    header = f"Group {int(group_id)} ({len(group_df)} records)"
                
                with st.expander(header):
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
    

    @staticmethod
    def render_duplicate_column_mapping(selected_file_ids, column_sets, file_options):
        """Render the column mapping interface for duplicate detection."""
        st.subheader("Column Mapping for Duplicate Detection")
        
        st.write("The selected files have different column structures. Please map columns to standardized names.")
        
        # Display column differences
        st.write("### Column differences:")
        for file_id, columns in column_sets:
            st.write(f"**{file_options[file_id]}**: {', '.join(sorted(list(columns)))}")
        
        # Create a unified set of all columns
        all_columns = set()
        for _, columns in column_sets:
            all_columns.update(columns)
        all_columns_list = sorted(list(all_columns))
        
        # Create mapping UI
        st.write("### Column Mapping:")
        
        # Target column names for first and second comparison columns
        col1, col2 = st.columns(2)
        with col1:
            first_col_target = st.text_input("Target name for first comparison column:", 
                                            value="first_name", 
                                            key="first_col_target")
        with col2:
            second_col_target = st.text_input("Target name for second comparison column:",
                                            value="last_name",
                                            key="second_col_target")
        
        # Create a mapping for each file
        file_mappings = {}
        for file_id, columns in column_sets:
            st.write(f"### Mapping for {file_options[file_id]}")
            
            # First column mapping
            first_col_options = ["-- Select Column --"] + sorted(list(columns))
            first_col_map = st.selectbox(
                f"Column for '{first_col_target}':",
                options=first_col_options,
                key=f"first_col_map_{file_id}"
            )
            
            # Second column mapping
            second_col_options = ["-- Select Column --"] + sorted(list(columns))
            second_col_map = st.selectbox(
                f"Column for '{second_col_target}':",
                options=second_col_options,
                key=f"second_col_map_{file_id}"
            )
            
            # Store mapping
            file_mappings[file_id] = {
                "first_col": first_col_map if first_col_map != "-- Select Column --" else None,
                "second_col": second_col_map if second_col_map != "-- Select Column --" else None
            }
        
        # Apply mapping button
        apply_mapping = st.button("Apply Mapping and Detect Duplicates", key="apply_column_mapping")
        
        # Validate mappings
        valid_mapping = True
        for file_id, mapping in file_mappings.items():
            if mapping["first_col"] is None or mapping["second_col"] is None:
                valid_mapping = False
                break
        
        if not valid_mapping and apply_mapping:
            st.error("Please select mappings for all files before proceeding.")
            apply_mapping = False
        
        return apply_mapping, file_mappings, first_col_target, second_col_target