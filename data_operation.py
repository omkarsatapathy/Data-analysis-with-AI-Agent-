import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union


class DataFrameOperations:
    """
    Handles operations on pandas DataFrames like merge, join, concat, etc.
    
    This class provides methods to perform various data manipulation operations
    on pandas DataFrames. It handles complex operations such as merging, joining,
    concatenation, handling missing values, pivoting, and aggregation while providing
    detailed explanations of the operations performed.
    """
    @staticmethod
    def perform_operation(
        operation_type: str, 
        params: Dict[str, Any], 
        dataframes: List[pd.DataFrame], 
        df_names: List[str]
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Perform a specified DataFrame operation.
        
        Args:
            operation_type (str): Type of operation (merge, join, concat, etc.)
            params (dict): Parameters for the operation
            dataframes (list): List of DataFrames to operate on
            df_names (list): Names of the DataFrames for documentation
            
        Returns:
            tuple: (resulting DataFrame, operation description) or (None, error message)
        """
        if len(dataframes) == 0:
            return None, "No DataFrames selected for operation"
        
        try:
            if operation_type == "merge":
                return DataFrameOperations._perform_merge(dataframes, df_names, params)
            elif operation_type == "concat":
                return DataFrameOperations._perform_concat(dataframes, df_names, params)
            elif operation_type == "fillna":
                return DataFrameOperations._perform_fillna(dataframes[0], df_names[0], params)
            elif operation_type == "dropna":
                return DataFrameOperations._perform_dropna(dataframes[0], df_names[0], params)
            elif operation_type == "melt":
                return DataFrameOperations._perform_melt(dataframes[0], df_names[0], params)
            elif operation_type == "pivot":
                return DataFrameOperations._perform_pivot(dataframes[0], df_names[0], params)
            elif operation_type == "groupby":
                return DataFrameOperations._perform_groupby(dataframes[0], df_names[0], params)
            else:
                return None, f"Unsupported operation: {operation_type}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return None, f"Error performing {operation_type} operation: {str(e)}\n\nDetails:\n{error_details}"
    
    @staticmethod
    def _perform_merge(
        dataframes: List[pd.DataFrame], 
        df_names: List[str], 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform merge operation between two DataFrames.
        
        Args:
            dataframes (list): List of DataFrames (expects at least 2)
            df_names (list): Names of the DataFrames
            params (dict): Parameters for the merge operation
            
        Returns:
            tuple: (merged DataFrame, operation description)
        """
        left_df = dataframes[0]
        right_df = dataframes[1] if len(dataframes) > 1 else dataframes[0]
        
        # Get merge parameters
        left_on = params.get('left_on')
        right_on = params.get('right_on', left_on)
        how = params.get('how', 'inner')
        simple_mode = params.get('simple_mode', False)
        
        # For simple mode (column fetch)
        if simple_mode:
            # Get fetch columns
            fetch_columns = params.get('fetch_columns', [])
            handle_dupes = params.get('handle_dupes', 'Keep first occurrence only')
            
            # Check for and handle duplicates if needed
            if handle_dupes in ['Keep first occurrence only', 'Keep last occurrence only']:
                # Deduplicate the right dataframe on the join key
                keep_option = 'first' if handle_dupes == 'Keep first occurrence only' else 'last'
                right_df = right_df.drop_duplicates(subset=[right_on], keep=keep_option)
                
                # Also deduplicate left dataframe if inner join to prevent duplicates
                if how == 'inner':
                    left_df = left_df.drop_duplicates(subset=[left_on], keep=keep_option)
            
            # Create a right dataframe with only the columns we need
            if fetch_columns:
                columns_to_fetch = [right_on] + fetch_columns
                right_df_subset = right_df[columns_to_fetch]
            else:
                right_df_subset = right_df
            
            # Perform the join
            result_df = pd.merge(
                left_df,
                right_df_subset,
                left_on=left_on,
                right_on=right_on,
                how=how
            )
            
            # Remove the duplicate ID column from the right dataframe
            if right_on in result_df.columns and f"{right_on}_x" in result_df.columns:
                # Rename columns to avoid _x and _y suffixes
                result_df = result_df.rename(columns={f"{right_on}_x": right_on})
                # Drop the duplicate column
                if f"{right_on}_y" in result_df.columns:
                    result_df = result_df.drop(columns=[f"{right_on}_y"])
            
            # Generate a user-friendly description
            fetch_desc = f"Fetched {len(fetch_columns)} columns" if fetch_columns else "Fetched all columns"
            how_desc = "Kept all rows from first file" if how == "left" else "Kept only matching rows"
            dedup_desc = ""
            if handle_dupes != "Keep all (may duplicate rows)":
                dedup_desc = f"Removed duplicate IDs (kept {handle_dupes.lower()})"
            
            description = f"""
            # Column Fetch Operation
            
            ## What happened
            - Added columns from **{df_names[1]}** to **{df_names[0]}**
            - Matched on: **{left_on}** = **{right_on}**
            - {how_desc}
            - {fetch_desc}
            {f"- {dedup_desc}" if dedup_desc else ""}
            
            ## Columns fetched
            {', '.join(['**' + col + '**' for col in fetch_columns]) if fetch_columns else 'All available columns'}
            
            ## Result details
            - Original files: {df_names[0]} ({dataframes[0].shape[0]} rows) and {df_names[1]} ({dataframes[1].shape[0]} rows)
            - Result: {result_df.shape[0]} rows × {result_df.shape[1]} columns
            
            ## Resulting columns
            {list(result_df.columns)}
            """
        else:
            # Full join mode
            # New way - explicit fetch columns
            fetch_columns = params.get('fetch_columns', [])
            
            # Old way - right_cols
            right_cols = params.get('right_cols', [])
            
            # If fetch_columns is specified (new method), use those
            if fetch_columns:
                # Always include the join column
                columns_to_include = [right_on] + fetch_columns
                # Filter the right dataframe 
                filtered_right_df = right_df[columns_to_include]
                
                # Perform merge
                result_df = pd.merge(
                    left_df, 
                    filtered_right_df,
                    left_on=left_on,
                    right_on=right_on,
                    how=how
                )
                
                # Generate description in a more user-friendly way
                description = f"""
                # Merge Operation
                
                ## What happened
                - Joined **{df_names[0]}** with **{df_names[1]}**
                - Matched on: **{left_on}** = **{right_on}**
                - Join type: **{how}**
                
                ## Columns fetched from {df_names[1]}
                {', '.join(['**' + col + '**' for col in fetch_columns])}
                
                ## Result details
                - Original shape: {df_names[0]} {dataframes[0].shape}, {df_names[1]} {right_df.shape}
                - Result shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns
                
                ## Resulting columns
                {list(result_df.columns)}
                """
            
            # Otherwise use the older right_cols method
            elif right_cols:
                # Always include the join column if it's not in the selected columns
                if right_on not in right_cols:
                    right_cols = [right_on] + right_cols
                
                # Filter the right dataframe to only include selected columns
                filtered_right_df = right_df[right_cols]
                
                # Perform merge
                result_df = pd.merge(
                    left_df, 
                    filtered_right_df,
                    left_on=left_on,
                    right_on=right_on,
                    how=how
                )
                
                # Generate description
                included_cols_info = "All columns" if not right_cols else f"Selected columns: {right_cols}"
                
                description = f"""
                # Merge Operation
                
                - Left DataFrame: {df_names[0]} (shape: {dataframes[0].shape})
                - Right DataFrame: {df_names[1]} (shape: {right_df.shape})
                - Merge type: {how}
                - Left join key: {left_on}
                - Right join key: {right_on}
                - Right DataFrame columns included: {included_cols_info}
                - Result shape: {result_df.shape}
                
                ## Common columns before merge:
                {set(left_df.columns).intersection(set(right_df.columns))}
                
                ## Resulting columns:
                {list(result_df.columns)}
                """
            
            # If neither is specified, do a regular merge with all columns
            else:
                # Perform merge with all columns
                result_df = pd.merge(
                    left_df, 
                    right_df,
                    left_on=left_on,
                    right_on=right_on,
                    how=how
                )
                
                # Generate description
                description = f"""
                # Merge Operation
                
                - Left DataFrame: {df_names[0]} (shape: {dataframes[0].shape})
                - Right DataFrame: {df_names[1]} (shape: {right_df.shape})
                - Merge type: {how}
                - Left join key: {left_on}
                - Right join key: {right_on}
                - Right DataFrame columns included: All columns
                - Result shape: {result_df.shape}
                
                ## Common columns before merge:
                {set(left_df.columns).intersection(set(right_df.columns))}
                
                ## Resulting columns:
                {list(result_df.columns)}
                """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    @staticmethod
    def _perform_concat(
        dataframes: List[pd.DataFrame], 
        df_names: List[str], 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform concatenation operation on DataFrames.
        
        Args:
            dataframes (list): List of DataFrames to concatenate
            df_names (list): Names of the DataFrames
            params (dict): Parameters for the concat operation
            
        Returns:
            tuple: (concatenated DataFrame, operation description)
        """
        axis = params.get('axis', 0)
        ignore_index = params.get('ignore_index', True)
        
        # Perform concatenation
        result_df = pd.concat(dataframes, axis=axis, ignore_index=ignore_index)
        
        description = f"""
        # Concatenation Operation
        
        - DataFrames: {', '.join(df_names)}
        - Axis: {'rows (0)' if axis == 0 else 'columns (1)'}
        - Ignore index: {ignore_index}
        - Original shapes: {', '.join([str(df.shape) for df in dataframes])}
        - Result shape: {result_df.shape}
        """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    @staticmethod
    def _perform_fillna(
        df: pd.DataFrame, 
        df_name: str, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform fillna operation on a DataFrame.
        
        Args:
            df (DataFrame): DataFrame to fill NA values in
            df_name (str): Name of the DataFrame
            params (dict): Parameters for the fillna operation
            
        Returns:
            tuple: (processed DataFrame, operation description)
        """
        value = params.get('value')
        method = params.get('method')
        
        if method:
            # Fill using method like 'ffill' or 'bfill'
            result_df = df.fillna(method=method)
            fill_desc = f"method='{method}'"
        else:
            # Fill using value
            result_df = df.fillna(value)
            fill_desc = f"value={value}"
        
        # Count NAs before and after
        na_before = df.isna().sum().sum()
        na_after = result_df.isna().sum().sum()
        
        description = f"""
        # Fill NA Operation
        
        - DataFrame: {df_name} (shape: {df.shape})
        - Fill parameters: {fill_desc}
        - NAs before: {na_before}
        - NAs after: {na_after}
        - NAs filled: {na_before - na_after}
        """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    @staticmethod
    def _perform_dropna(
        df: pd.DataFrame, 
        df_name: str, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform dropna operation on a DataFrame.
        
        Args:
            df (DataFrame): DataFrame to drop NA values from
            df_name (str): Name of the DataFrame
            params (dict): Parameters for the dropna operation
            
        Returns:
            tuple: (processed DataFrame, operation description)
        """
        axis = params.get('axis', 0)
        how = params.get('how', 'any')
        
        # Perform drop NA
        result_df = df.dropna(axis=axis, how=how)
        
        description = f"""
        # Drop NA Operation
        
        - DataFrame: {df_name}
        - Axis: {'rows (0)' if axis == 0 else 'columns (1)'}
        - How: {how}
        - Shape before: {df.shape}
        - Shape after: {result_df.shape}
        - {'Rows' if axis == 0 else 'Columns'} removed: {df.shape[axis] - result_df.shape[axis]}
        """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    @staticmethod
    def _perform_melt(
        df: pd.DataFrame, 
        df_name: str, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform melt operation on a DataFrame.
        
        Args:
            df (DataFrame): DataFrame to melt
            df_name (str): Name of the DataFrame
            params (dict): Parameters for the melt operation
            
        Returns:
            tuple: (melted DataFrame, operation description)
        """
        id_vars = params.get('id_vars', [])
        value_vars = params.get('value_vars', [col for col in df.columns if col not in id_vars])
        
        # Perform melt
        result_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=params.get('var_name', 'variable'),
            value_name=params.get('value_name', 'value')
        )
        
        description = f"""
        # Melt Operation
        
        - DataFrame: {df_name}
        - ID variables: {id_vars}
        - Value variables: {len(value_vars)} columns
        - Shape before: {df.shape}
        - Shape after: {result_df.shape}
        """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    @staticmethod
    def _perform_pivot(
        df: pd.DataFrame, 
        df_name: str, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform pivot operation on a DataFrame.
        
        Args:
            df (DataFrame): DataFrame to pivot
            df_name (str): Name of the DataFrame
            params (dict): Parameters for the pivot operation
            
        Returns:
            tuple: (pivoted DataFrame, operation description)
        """
        index = params.get('index')
        columns = params.get('columns')
        values = params.get('values')
        
        # Perform pivot
        if values:
            result_df = df.pivot(index=index, columns=columns, values=values)
        else:
            result_df = df.pivot(index=index, columns=columns)
        
        description = f"""
        # Pivot Operation
        
        - DataFrame: {df_name}
        - Index: {index}
        - Columns: {columns}
        - Values: {values if values else 'All other columns'}
        - Shape before: {df.shape}
        - Shape after: {result_df.shape}
        """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description
    
    @staticmethod
    def _perform_groupby(
        df: pd.DataFrame, 
        df_name: str, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Perform groupby operation on a DataFrame.
        
        Args:
            df (DataFrame): DataFrame to group
            df_name (str): Name of the DataFrame
            params (dict): Parameters for the groupby operation
            
        Returns:
            tuple: (grouped DataFrame, operation description)
        """
        by = params.get('by', [])
        agg = params.get('agg', {})
        
        # Perform groupby
        result_df = df.groupby(by).agg(agg).reset_index()
        
        description = f"""
        # Group By Operation
        
        - DataFrame: {df_name}
        - Grouping columns: {by}
        - Aggregations: {agg}
        - Shape before: {df.shape}
        - Shape after: {result_df.shape}
        - Number of groups: {len(result_df)}
        """
        
        # Add DataFrame summary
        full_description = description + "\n" + DataAnalyzer.get_dataframe_summary(result_df, "Result")
        
        return result_df, full_description


class DataAnalyzer:
    """
    Provides methods for analyzing and reporting on data quality and structure.
    
    This class includes tools for generating data quality reports, examining
    DataFrame structures, and producing summary statistics for data exploration.
    """
    @staticmethod
    def generate_data_quality_report(df: pd.DataFrame) -> str:
        """
        Generate a comprehensive data quality report for the DataFrame.
        
        Args:
            df (DataFrame): The DataFrame to analyze
            
        Returns:
            str: A formatted data quality report
        """
        if not isinstance(df, pd.DataFrame):
            return "Error: No DataFrame to analyze"
        
        # Basic information
        row_count = df.shape[0]
        column_count = df.shape[1]
        
        # NaN information
        nan_counts = df.isna().sum()
        non_nan_counts = df.notna().sum()
        
        # Calculate percentages
        if row_count > 0:
            nan_percentages = (nan_counts / row_count * 100).round(2)
        else:
            nan_percentages = nan_counts * 0
        
        # Find column with highest percentage of NaNs
        if not nan_percentages.empty and nan_percentages.max() > 0:
            max_nan_col = nan_percentages.idxmax()
            max_nan_percentage = nan_percentages.max()
        else:
            max_nan_col = "None"
            max_nan_percentage = 0
        
        # Data type information
        dtypes = df.dtypes
        
        # Special check for phone numbers
        phone_number_cols = []
        phone_quality_info = []
        
        # Look for columns that might contain phone numbers
        for col in df.columns:
            col_lower = str(col).lower()
            if any(phone_term in col_lower for phone_term in ["phone", "mobile", "cell", "tel", "contact"]):
                phone_number_cols.append(col)
        
        # Analyze phone number columns if any
        for col in phone_number_cols:
            # Skip if column has all NaN values
            if nan_counts[col] == row_count:
                continue
            
            # Only analyze string columns or columns that could contain phone numbers
            if df[col].dtype == 'object' or 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
                # Convert to string and clean
                phone_data = df[col].fillna('').astype(str)
                
                # Count digits in each phone number
                digit_counts = phone_data.apply(lambda x: sum(c.isdigit() for c in x))
                
                # Basic phone validation (only count non-empty values)
                non_empty = (phone_data != '').sum()
                valid_10_digit = ((digit_counts == 10) & (phone_data.str.len() >= 10)).sum()
                valid_with_country_code = ((digit_counts > 10) & (digit_counts <= 15)).sum()
                invalid_phones = non_empty - valid_10_digit - valid_with_country_code
                
                if non_empty > 0:
                    phone_info = f"""
                Column '{col}' phone number analysis:
                - Valid 10-digit numbers: {valid_10_digit} ({valid_10_digit/non_empty*100:.2f}% of non-empty values)
                - Valid with country code: {valid_with_country_code} ({valid_with_country_code/non_empty*100:.2f}% of non-empty values)
                - Invalid or unusual formats: {invalid_phones} ({invalid_phones/non_empty*100:.2f}% of non-empty values)
                    """
                    phone_quality_info.append(phone_info)
        
        # Build the report
        report = f"""
        # DATA QUALITY REPORT
        
        ## Basic Information
        - Total rows: {row_count:,}
        - Total columns: {column_count}
        
        ## Missing Data Summary
        - Total missing values: {nan_counts.sum():,}
        - Column with most missing values: '{max_nan_col}' ({max_nan_percentage:.2f}% missing)
        
        ## Column-by-Column Analysis
        """
        
        # Add column details
        for col in df.columns:
            non_null_percent = non_nan_counts[col]/row_count*100 if row_count > 0 else 0
            report += f"""
        '{col}':
          - Data type: {dtypes[col]}
          - Non-null values: {non_nan_counts[col]:,} ({non_null_percent:.2f}%)
          - Missing values: {nan_counts[col]:,} ({nan_percentages[col]:.2f}%)
            """
            
            # For numeric columns, add basic stats
            if pd.api.types.is_numeric_dtype(df[col]):
                if non_nan_counts[col] > 0:
                    try:
                        report += f"""
          - Min: {df[col].min()}
          - Max: {df[col].max()}
          - Mean: {df[col].mean():.2f}
          - Unique values: {df[col].nunique():,}
                        """
                    except:
                        # In case of any calculation errors
                        report += f"""
          - Unable to calculate statistics for this column
                        """
            # For categorical/text columns
            elif df[col].dtype == 'object':
                if non_nan_counts[col] > 0:
                    try:
                        value_counts = df[col].value_counts()
                        report += f"""
          - Unique values: {df[col].nunique():,} ({df[col].nunique()/row_count*100:.2f}% of total rows)
                        """
                        if not value_counts.empty:
                            report += f"""
          - Most common value: '{value_counts.index[0]}' (occurs {value_counts.iloc[0]:,} times)
                            """
                    except:
                        report += f"""
          - Unable to analyze categorical values
                        """
        
        # Add phone number quality info if any
        if phone_quality_info:
            report += "\n        ## Phone Number Quality Analysis\n"
            for info in phone_quality_info:
                report += info
        
        # Add data quality score
        # Simple score based on completeness (100% - average percentage of missing values)
        avg_missing_percentage = nan_percentages.mean() if not nan_percentages.empty else 0
        quality_score = 100 - avg_missing_percentage
        
        report += f"""
        ## Overall Data Quality Score
        - Completeness score: {quality_score:.2f}/100
        """
        
        # Add duplicate check
        try:
            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / row_count * 100) if row_count > 0 else 0
            report += f"""
        ## Duplicate Analysis
        - Duplicate rows: {duplicate_rows:,} ({duplicate_percentage:.2f}% of total rows)
            """
        except:
            report += """
        ## Duplicate Analysis
        - Unable to check for duplicates
            """
        
        return report
    
    @staticmethod
    def get_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame") -> str:
        """
        Generate a comprehensive summary of a DataFrame.
        
        Args:
            df (DataFrame): The DataFrame to summarize
            name (str): Name to use for the DataFrame in the summary
            
        Returns:
            str: A formatted summary of the DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            return f"Error: {name} is not a valid DataFrame"
        
        # Basic information
        summary = f"""
        # {name} Summary
        
        ## Basic Information
        - Shape: {df.shape[0]} rows × {df.shape[1]} columns
        - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        
        ## Column Information
        """
        
        # Column information
        for col in df.columns:
            dtype = df[col].dtype
            na_count = df[col].isna().sum()
            na_percent = (na_count / len(df) * 100) if len(df) > 0 else 0
            
            summary += f"""
        - {col} ({dtype})
          - Missing values: {na_count} ({na_percent:.2f}%)
        """
            
            # Add statistics based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].count() > 0:  # Only add stats if there are non-NA values
                    summary += f"""
          - Min: {df[col].min()}
          - Max: {df[col].max()}
          - Mean: {df[col].mean():.4f}
          - Std Dev: {df[col].std():.4f}
        """
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                unique_count = df[col].nunique()
                summary += f"""
          - Unique values: {unique_count}
        """
                if unique_count < 10 and unique_count > 0:  # Show value counts for low cardinality columns
                    value_counts = df[col].value_counts().head(5)
                    summary += "          - Top values:\n"
                    for val, count in value_counts.items():
                        pct = (count / df[col].count() * 100) if df[col].count() > 0 else 0
                        summary += f"            - {val}: {count} ({pct:.2f}%)\n"
        
        # Add head of DataFrame
        summary += f"""
        ## Data Preview (First 5 rows)
        {df.head().to_string()}
        """
        
        return summary