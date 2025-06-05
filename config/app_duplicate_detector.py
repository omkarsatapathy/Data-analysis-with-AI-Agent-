import streamlit as st
import pandas as pd
import uuid
import datetime
from typing import Tuple, Dict, Any, Optional, List
from src.file_export import FileExporter

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
                # Check for column mismatches if multiple files are selected
                if len(st.session_state.uploaded_files) > 1:
                    # Collect column sets from each file
                    column_sets = []
                    file_options = {}
                    
                    for file_id, file_info in st.session_state.uploaded_files.items():
                        column_sets.append((file_id, set(file_info['content'].columns)))
                        file_options[file_id] = file_info['name']
                    
                    # Check for column mismatches
                    has_mismatches = False
                    mismatch_details = []
                    
                    # Get a unified set of all columns across files
                    all_columns = set()
                    for _, columns in column_sets:
                        all_columns.update(columns)
                    
                    # Check each file for missing columns compared to the complete set
                    for file_id, columns in column_sets:
                        missing_columns = all_columns - columns
                        if missing_columns:
                            has_mismatches = True
                            mismatch_details.append({
                                'file': file_options[file_id],
                                'missing': sorted(list(missing_columns))
                            })
                    
                    # Display appropriate warning based on column mismatches
                    if has_mismatches:
                        st.warning("""
                            ⚠️ **Column mismatches detected!** 
                            The selected files have different column structures, which will cause problems 
                            with duplicate detection. Please standardize column names before proceeding.
                        """)
                        
                        # Show mismatch details
                        with st.expander("View column mismatches", expanded=True):
                            for mismatch in mismatch_details:
                                st.write(f"**{mismatch['file']}** is missing columns: {', '.join(mismatch['missing'])}")
                    else:
                        st.success("""
                            ✓ **Column structures match across files.** 
                            All selected files have the same column names.
                        """)
                
                # Let user specify if they want to convert all columns to string
                convert_to_string = st.checkbox("Convert all columns to string before checking duplicates", value=True)
                
                # Let user select similarity threshold - MOVED THIS EARLIER in the function
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
                
                # File selection for multiple files
                selected_file_ids = []
                if len(st.session_state.uploaded_files) > 1:
                    st.write("### Select Files for Duplicate Detection")
                    file_options = {file_id: file_info['name'] for file_id, file_info in st.session_state.uploaded_files.items()}
                    
                    selected_file_ids = st.multiselect(
                        "Select files to check for duplicates:",
                        options=list(file_options.keys()),
                        default=list(file_options.keys()),
                        format_func=lambda x: file_options[x],
                        key="duplicate_file_selector"
                    )
                else:
                    # If only one file, pre-select it
                    selected_file_ids = [st.session_state.active_file_id] if st.session_state.active_file_id else []
                
                # Show all files with their columns for reference
                if len(st.session_state.uploaded_files) > 1:
                    with st.expander("View columns in all files", expanded=True):
                        for file_id, file_info in st.session_state.uploaded_files.items():
                            st.write(f"**{file_info['name']}**")
                            st.write(f"Columns: {', '.join(file_info['content'].columns.tolist())}")
                            st.write("---")
                
                # Additional warning about selected columns if multiple files are chosen
                if len(selected_file_ids) > 1 and (first_col or second_col):
                    # Check if the selected columns exist in all selected files
                    missing_in_files = []
                    for file_id in selected_file_ids:
                        if file_id in st.session_state.uploaded_files:
                            file_df = st.session_state.uploaded_files[file_id]['content']
                            file_name = st.session_state.uploaded_files[file_id]['name']
                            
                            missing_cols = []
                            if first_col and first_col not in file_df.columns:
                                missing_cols.append(first_col)
                            if second_col and second_col not in file_df.columns:
                                missing_cols.append(second_col)
                                
                            if missing_cols:
                                missing_in_files.append((file_name, missing_cols))
                    
                    if missing_in_files:
                        # Show warning about missing columns in some files
                        warning_message = "⚠️ **Warning:** Selected columns missing in some files:\n"
                        for file_name, missing_cols in missing_in_files:
                            warning_message += f"- **{file_name}** missing: {', '.join(missing_cols)}\n"
                        warning_message += "\nPlease standardize column names before proceeding."
                        
                        st.warning(warning_message)
                        
                        # Disable the detect button if columns are missing
                        detect_button = st.button("Detect Duplicates", key="detect_duplicates_button", disabled=True)
                        if st.button("Standardize Column Names Now", key="fix_columns_now"):
                            # Store parameters for later - Now threshold is defined earlier!
                            st.session_state.duplicate_first_col = first_col
                            st.session_state.duplicate_second_col = second_col
                            st.session_state.duplicate_threshold = threshold
                            st.session_state.duplicate_convert_to_string = convert_to_string
                            st.session_state.column_transform_triggered_by_duplicate = True
                            
                            # Redirect to column transformation
                            st.session_state.show_column_transform_input = True
                            st.rerun()
                            return None, None, None, False, False, []
                    else:
                        # All good - the selected columns exist in all selected files
                        detect_button = st.button("Detect Duplicates", key="detect_duplicates_button")
                else:
                    # Normal case - just show the detect button
                    detect_button = st.button("Detect Duplicates", key="detect_duplicates_button")
                
                return first_col, second_col, threshold, convert_to_string, detect_button, selected_file_ids
            else:
                st.warning("No data loaded. Please upload a file first.")
        
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
            merged_df = st.session_state.merged_df
            if "_source_file" in duplicate_df.columns:
                st.write("### Files containing duplicates:")
                file_counts = duplicate_df['_source_file'].value_counts().reset_index()
                file_counts.columns = ['File', 'Records']
                st.dataframe(file_counts)
            
            # Show each group in an expander
            for group_id in sorted(duplicate_df['duplicate_group'].unique()):
                group_df = duplicate_df[duplicate_df['duplicate_group'] == group_id]
                
                # Ensure column names are unique before displaying
                group_df_display = group_df.copy()
                
                # Check for duplicate columns and make them unique if needed
                if group_df_display.columns.duplicated().any():
                    # Create unique column names by adding suffixes
                    cols = list(group_df_display.columns)
                    col_counts = {}
                    for i, col in enumerate(cols):
                        if col in col_counts:
                            col_counts[col] += 1
                            cols[i] = f"{col}_{col_counts[col]}"
                        else:
                            col_counts[col] = 0
                    
                    group_df_display.columns = cols
                
                # Create a more descriptive header
                if "_source_file" in group_df.columns:
                    file_count = group_df['_source_file'].nunique()
                    files_str = "from multiple files" if file_count > 1 else "in same file"
                    header = f"Group {int(group_id)} ({len(group_df)} records {files_str})"
                else:
                    header = f"Group {int(group_id)} ({len(group_df)} records)"
                
                with st.expander(header):
                    st.dataframe(group_df_display)       
            
            column1, column2 = st.columns(2)
            with column1:
                # Add download button
                if st.button("Download merged records without duplicates", key="download_merged"):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"merged_records_{timestamp}.csv"
                    
                    # Save to CSV
                    merged_df.to_csv(filename, index=False)
                    
                    # Add to generated files
                    file_id = str(uuid.uuid4())
                    st.session_state.generated_files[file_id] = {
                        'name': f"merged_records_{timestamp}.csv",
                        'path': filename,
                        'type': 'csv'
                    }
                    
                    st.success(f"File saved to {filename}")

                file_path = st.text_input("Enter file path: ")
                file_format = st.selectbox("Select file format: ", ["xlsx", "parquet", "csv"])
                if st.button("Export now", key="export_merged_recors"):
                    message, success = FileExporter.export_dataframe(merged_df, file_path, file_format)
                    if success:
                        st.success("File exported successfully")
                    else:
                        st.error(message)

            with column2:
                if st.button("Download Duplicate Records", key="download_duplicates"):
                    # Generate a filename with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"duplicate_records_{timestamp}.csv"
                    
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
                                            value="First name", 
                                            key="first_col_target")
        with col2:
            second_col_target = st.text_input("Target name for second comparison column:",
                                            value="Last name",
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