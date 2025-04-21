import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

class AppDataOperations:
    """
    Manages the data operation UI components.
    
    This class encapsulates UI elements for DataFrame operations like
    merge, concat, fillna, etc.
    """
    
    @staticmethod
    def render_dataframe_operations():
        """Render the DataFrame operations interface."""
        if st.session_state.show_df_operations:
            st.subheader("DataFrame Operations")
            
            if len(st.session_state.uploaded_files) > 0:
                # Create multiselect for selecting files
                file_options = {file_id: file_info['name'] for file_id, file_info in st.session_state.uploaded_files.items()}
                
                selected_file_ids = st.multiselect(
                    "Select files to operate on:",
                    options=list(file_options.keys()),
                    format_func=lambda x: file_options[x],
                    default=[st.session_state.active_file_id] if st.session_state.active_file_id else [],
                    key="operation_file_selector"
                )
                
                # Select operation type
                operation_type = st.selectbox(
                    "Select operation type:",
                    options=["merge", "concat", "fillna", "dropna", "melt", "pivot", "groupby"],
                    key="operation_type"
                )
                
                # Create container for parameters based on operation type
                params_container = st.container()
                params = {}
                
                with params_container:
                    if operation_type == "merge":
                        params = AppDataOperations._render_merge_params(selected_file_ids)
                    elif operation_type == "concat":
                        params = AppDataOperations._render_concat_params()
                    elif operation_type == "fillna":
                        params = AppDataOperations._render_fillna_params()
                    elif operation_type == "dropna":
                        params = AppDataOperations._render_dropna_params()
                    elif operation_type == "melt":
                        params = AppDataOperations._render_melt_params(selected_file_ids)
                    elif operation_type == "pivot":
                        params = AppDataOperations._render_pivot_params(selected_file_ids)
                    elif operation_type == "groupby":
                        params = AppDataOperations._render_groupby_params(selected_file_ids)
                
                # Execute operation button
                execute_button = st.button("Execute Operation", key="execute_operation_button")
                
                return selected_file_ids, operation_type, params, execute_button
            else:
                st.warning("No files loaded. Please upload files first.")
        
        return None, None, None, False
    
    @staticmethod
    def _render_merge_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for merge operation."""
        params = {}
        
        if len(selected_file_ids) < 2:
            st.warning("Merge operation requires at least 2 files. Please select another file.")
            return params
        
        # Get metadata for the files
        file1_name = st.session_state.uploaded_files[selected_file_ids[0]]['name']
        file2_name = st.session_state.uploaded_files[selected_file_ids[1]]['name']
        
        # Get dataframes for both files
        df1 = st.session_state.uploaded_files[selected_file_ids[0]]['content']
        df2 = st.session_state.uploaded_files[selected_file_ids[1]]['content']
        
        # Get column lists
        df1_columns = list(df1.columns)
        df2_columns = list(df2.columns)
        
        # Find common columns for join key suggestions
        common_columns = list(set(df1_columns) & set(df2_columns))
        
        # Default to "ID" column if it exists, otherwise use first common column
        default_id_col = None
        for col_name in ['ID', 'Id', 'id', 'iD']:
            if col_name in common_columns:
                default_id_col = col_name
                break
        
        if default_id_col is None and common_columns:
            default_id_col = common_columns[0]
        elif default_id_col is None:
            default_id_col = df1_columns[0] if df1_columns else None
        
        # Let user choose between simple column fetch or full join
        operation_subtype = st.radio(
            "Choose operation type:",
            options=["Simple column fetch (add columns from second file)", 
                    "Full join (SQL-style merge with all options)"],
            index=0,
            key="operation_subtype"
        )
        
        if operation_subtype == "Simple column fetch (add columns from second file)":
            # Simple column fetch UI
            st.write(f"### Add columns from {file2_name} to {file1_name}")
            
            # First: Ask what ID column to match on in each file
            st.write("**Step 1:** Select the ID columns to match")
            
            col1a, col1b = st.columns(2)
            with col1a:
                params['left_on'] = st.selectbox(
                    f"ID column in {file1_name}:",
                    options=df1_columns,
                    index=df1_columns.index(default_id_col) if default_id_col in df1_columns else 0,
                    key="left_on"
                )
            
            with col1b:
                params['right_on'] = st.selectbox(
                    f"ID column in {file2_name}:",
                    options=df2_columns,
                    index=df2_columns.index(default_id_col) if default_id_col in df2_columns else 0,
                    key="right_on"
                )
            
            # Check for duplicate keys in both dataframes
            left_dupes = df1[params['left_on']].duplicated().any()
            right_dupes = df2[params['right_on']].duplicated().any()
            
            if left_dupes or right_dupes:
                st.warning(f"⚠️ Warning: Duplicate IDs detected! " + 
                        (f"'{params['left_on']}' has duplicates in {file1_name}. " if left_dupes else "") +
                        (f"'{params['right_on']}' has duplicates in {file2_name}. " if right_dupes else "") +
                        "This may cause row duplication in the result.")
                
                # Offer option to handle duplicates
                handle_dupes = st.selectbox(
                    "How to handle duplicates:",
                    options=["Keep first occurrence only", "Keep last occurrence only", "Keep all (may duplicate rows)"],
                    index=0,
                    key="handle_dupes"
                )
                params['handle_dupes'] = handle_dupes
            
            # Second: Ask what columns to fetch from the second file
            st.write("**Step 2:** Select columns to fetch from the second table")
            
            # Exclude the join column from the fetch options to avoid duplicates
            fetch_options = [col for col in df2_columns if col != params['right_on']]
            
            # Allow selecting columns to fetch from second table
            params['fetch_columns'] = st.multiselect(
                f"Select columns to fetch from {file2_name}:",
                options=fetch_options,
                key="fetch_columns"
            )
            
            # If nothing selected, offer to fetch all remaining columns
            if not params['fetch_columns']:
                fetch_all = st.checkbox("Fetch all columns", value=True, key="fetch_all")
                if fetch_all:
                    params['fetch_columns'] = fetch_options
            
            # What to do with non-matching rows
            params['how'] = st.radio(
                "How to handle rows that don't have a match:",
                options=["Keep rows with missing values", "Only keep rows with matches"],
                index=0,
                key="missing_handling",
                help="'Keep rows with missing values' is like LEFT JOIN, 'Only keep rows with matches' is like INNER JOIN"
            )
            
            # Convert the user-friendly options to the actual join type
            if params['how'] == "Keep rows with missing values":
                params['how'] = "left"
            else:
                params['how'] = "inner"
            
            # Simple mode is just column fetch
            params['simple_mode'] = True
            
        else:
            # Full join UI
            st.write(f"### Full Join Operation: {file1_name} with {file2_name}")
            
            # Join key selection
            col1a, col1b = st.columns(2)
            with col1a:
                params['left_on'] = st.selectbox(
                    f"Left join key column ({file1_name}):",
                    options=df1_columns,
                    index=df1_columns.index(default_id_col) if default_id_col in df1_columns else 0,
                    key="left_on"
                )
            
            with col1b:
                # If we have common columns, default to the same column name
                # Otherwise, let the user pick any column from the right dataframe
                right_default_index = df2_columns.index(params['left_on']) if params['left_on'] in df2_columns else 0
                params['right_on'] = st.selectbox(
                    f"Right join key column ({file2_name}):",
                    options=df2_columns,
                    index=right_default_index,
                    key="right_on"
                )
            
            # Select join type
            params['how'] = st.selectbox(
                "Join type:",
                options=["left", "inner", "right", "outer"],
                index=0,  # Default to left join
                key="join_type"
            )
            
            # Allow the user to select which columns to include from the right dataframe
            st.write("Select columns to include from the right dataframe:")
            
            # Add 'All columns' option
            all_cols_option = "[All columns]"
            right_cols_options = [all_cols_option] + df2_columns
            
            selected_right_cols = st.multiselect(
                "Columns to include (leave empty for all):",
                options=right_cols_options,
                default=[all_cols_option],
                key="right_cols_to_include"
            )
            
            # Store in params
            if not selected_right_cols or all_cols_option in selected_right_cols:
                params['right_cols'] = None  # All columns
            else:
                params['right_cols'] = selected_right_cols
            
            # Full mode - not just column fetch
            params['simple_mode'] = False
        
        return params
    
    @staticmethod
    def _render_concat_params() -> Dict[str, Any]:
        """Render and collect parameters for concat operation."""
        params = {}
        
        # Axis selection
        params['axis'] = st.radio(
            "Select concatenation direction:",
            options=["Rows (vertically)", "Columns (horizontally)"],
            index=0,
            key="concat_axis"
        )
        
        # Convert to 0 for rows, 1 for columns
        params['axis'] = 0 if params['axis'] == "Rows (vertically)" else 1
        
        # Ignore index option
        params['ignore_index'] = st.checkbox(
            "Reset index in result",
            value=True,
            key="concat_ignore_index",
            help="If checked, the original indices will be discarded and replaced with a new sequential index"
        )
        
        return params
    
    @staticmethod
    def _render_fillna_params() -> Dict[str, Any]:
        """Render and collect parameters for fillna operation."""
        params = {}
        
        # Fill method
        fill_method = st.radio(
            "Select fill method:",
            options=["Fill with specific value", "Fill with statistics", "Fill with interpolation"],
            index=0,
            key="fillna_method"
        )
        
        if fill_method == "Fill with specific value":
            # Fill value
            fill_value = st.text_input(
                "Fill value (leave empty for NaN):",
                key="fillna_value"
            )
            
            # Try to convert to appropriate type
            try:
                if fill_value.strip() == "":
                    fill_value = None
                elif fill_value.lower() == "true":
                    fill_value = True
                elif fill_value.lower() == "false":
                    fill_value = False
                elif fill_value.isdigit():
                    fill_value = int(fill_value)
                else:
                    try:
                        fill_value = float(fill_value)
                    except ValueError:
                        # Keep as string
                        pass
            except:
                # Keep as is if any error
                pass
            
            params['method'] = 'value'
            params['value'] = fill_value
            
        elif fill_method == "Fill with statistics":
            # Statistical method
            stat_method = st.selectbox(
                "Fill with:",
                options=["mean", "median", "mode"],
                index=0,
                key="fillna_stat_method"
            )
            
            params['method'] = stat_method
            
        else:  # Interpolation
            # Interpolation method
            interp_method = st.selectbox(
                "Interpolation method:",
                options=["linear", "pad", "nearest", "polynomial"],
                index=0,
                key="fillna_interp_method"
            )
            
            params['method'] = 'interpolate'
            params['interp_method'] = interp_method
        
        return params
    
    @staticmethod
    def _render_dropna_params() -> Dict[str, Any]:
        """Render and collect parameters for dropna operation."""
        params = {}
        
        # Drop axis
        params['axis'] = st.radio(
            "Drop rows or columns with missing values:",
            options=["Drop rows", "Drop columns"],
            index=0,
            key="dropna_axis"
        )
        
        # Convert to 0 for rows, 1 for columns
        params['axis'] = 0 if params['axis'] == "Drop rows" else 1
        
        # How option
        params['how'] = st.radio(
            "When to drop:",
            options=["Drop if any value is missing", "Drop only if all values are missing"],
            index=0,
            key="dropna_how"
        )
        
        # Convert to 'any' or 'all'
        params['how'] = 'any' if params['how'] == "Drop if any value is missing" else 'all'
        
        # Threshold option
        params['thresh'] = st.number_input(
            "Minimum number of non-NA values to keep (0 = no threshold):",
            min_value=0,
            value=0,
            step=1,
            key="dropna_thresh"
        )
        
        # Convert 0 to None for thresh
        if params['thresh'] == 0:
            params['thresh'] = None
        
        return params
    
    @staticmethod
    def _render_melt_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for melt operation."""
        params = {}
        
        if not selected_file_ids:
            st.warning("Please select a file to melt.")
            return params
        
        # Get the DataFrame to melt
        df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
        columns = list(df.columns)
        
        # ID columns (not to be melted)
        params['id_vars'] = st.multiselect(
            "Select ID columns (columns to keep):",
            options=columns,
            key="melt_id_vars"
        )
        
        # Value columns (to be melted)
        remaining_columns = [col for col in columns if col not in params['id_vars']]
        params['value_vars'] = st.multiselect(
            "Select value columns to melt (leave empty to melt all non-ID columns):",
            options=remaining_columns,
            key="melt_value_vars"
        )
        
        # If no value columns selected, use all remaining columns
        if not params['value_vars']:
            params['value_vars'] = remaining_columns
        
        # Column names for melted data
        col1, col2 = st.columns(2)
        with col1:
            params['var_name'] = st.text_input(
                "Variable column name:",
                value="variable",
                key="melt_var_name"
            )
        
        with col2:
            params['value_name'] = st.text_input(
                "Value column name:",
                value="value",
                key="melt_value_name"
            )
        
        return params
    
    @staticmethod
    def _render_pivot_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for pivot operation."""
        params = {}
        
        if not selected_file_ids:
            st.warning("Please select a file to pivot.")
            return params
        
        # Get the DataFrame to pivot
        df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
        columns = list(df.columns)
        
        # Column selections
        col1, col2, col3 = st.columns(3)
        
        with col1:
            params['index'] = st.selectbox(
                "Select index column:",
                options=columns,
                key="pivot_index"
            )
        
        with col2:
            # Filter out the index column
            column_options = [col for col in columns if col != params['index']]
            params['columns'] = st.selectbox(
                "Select column headers:",
                options=column_options,
                key="pivot_columns"
            )
        
        with col3:
            # Filter out index and column columns
            value_options = [col for col in columns if col not in [params['index'], params['columns']]]
            params['values'] = st.selectbox(
                "Select values column:",
                options=value_options,
                key="pivot_values"
            )
        
        # Aggregation function
        params['aggfunc'] = st.selectbox(
            "Select aggregation function (if multiple values):",
            options=["mean", "sum", "count", "min", "max", "first", "last"],
            index=0,
            key="pivot_aggfunc"
        )
        
        # Fill NaN values
        fill_na = st.checkbox(
            "Fill missing values with:",
            value=False,
            key="pivot_fill_na_check"
        )
        
        if fill_na:
            fill_value = st.number_input(
                "Fill value:",
                value=0,
                key="pivot_fill_value"
            )
            params['fill_value'] = fill_value
        else:
            params['fill_value'] = None
        
        return params
    
    @staticmethod
    def _render_groupby_params(selected_file_ids: List[str]) -> Dict[str, Any]:
        """Render and collect parameters for groupby operation."""
        params = {}
        
        if not selected_file_ids:
            st.warning("Please select a file to group.")
            return params
        
        # Get the DataFrame to group
        df = st.session_state.uploaded_files[selected_file_ids[0]]['content']
        columns = list(df.columns)
        
        # Group by columns
        params['by'] = st.multiselect(
            "Select columns to group by:",
            options=columns,
            key="groupby_by"
        )
        
        if not params['by']:
            st.warning("Please select at least one column to group by.")
            return params
        
        # Select columns to aggregate
        remaining_columns = [col for col in columns if col not in params['by']]
        agg_columns = st.multiselect(
            "Select columns to aggregate (leave empty to aggregate all non-groupby columns):",
            options=remaining_columns,
            key="groupby_agg_columns"
        )
        
        # If no aggregation columns, use all remaining columns
        if not agg_columns:
            agg_columns = remaining_columns
        
        # Aggregation functions per column
        st.write("Select aggregation functions for each column:")
        
        agg_funcs = {}
        for col in agg_columns:
            funcs = st.multiselect(
                f"Aggregation functions for '{col}':",
                options=["mean", "sum", "count", "min", "max", "first", "last", "std", "var"],
                default=["mean"],
                key=f"groupby_agg_{col}"
            )
            
            if funcs:
                agg_funcs[col] = funcs
        
        params['agg_funcs'] = agg_funcs
        
        # Include reset index option
        params['reset_index'] = st.checkbox(
            "Reset index after groupby",
            value=True,
            key="groupby_reset_index"
        )
        
        return params