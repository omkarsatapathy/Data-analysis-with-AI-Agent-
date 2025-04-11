import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


class ColumnTransformer:
    """
    Handles column transformations on pandas DataFrames, particularly column renaming
    based on mapping files.
    
    This class provides methods to load column mapping files, validate their structure,
    apply column transformations to DataFrames, and generate detailed transformation reports.
    """
    
    @staticmethod
    def load_mapping_file(mapping_file) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load and validate a column mapping file.
        
        Args:
            mapping_file: Uploaded file object containing column mappings
            
        Returns:
            tuple: (DataFrame with mappings or None if invalid, status message)
        """
        try:
            # Determine file type from extension
            file_name = mapping_file.name.lower()
            
            if file_name.endswith('.csv'):
                mapping_df = pd.read_csv(mapping_file)
            elif file_name.endswith(('.xlsx', '.xls')):
                mapping_df = pd.read_excel(mapping_file)
            else:
                return None, "Unsupported file format. Please upload a CSV or Excel file."
            
            # Validate mapping file structure
            if mapping_df.shape[1] < 2:
                return None, "Mapping file must have at least 2 columns for old and new column names."
            
            # Ensure the first two columns are for mapping (regardless of their actual names)
            mapping_df = mapping_df.iloc[:, :2]
            
            # Rename the first two columns for consistency
            mapping_df.columns = ['old_column_name', 'new_column_name']
            
            # Drop rows with missing values
            mapping_df = mapping_df.dropna()
            
            # Ensure column names are strings
            mapping_df['old_column_name'] = mapping_df['old_column_name'].astype(str)
            mapping_df['new_column_name'] = mapping_df['new_column_name'].astype(str)
            
            return mapping_df, "Mapping file loaded successfully."
            
        except Exception as e:
            return None, f"Error loading mapping file: {str(e)}"
    
    @staticmethod
    def transform_columns(
        df: pd.DataFrame, 
        mapping_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform column names in a DataFrame based on the provided mapping.
        
        Args:
            df: DataFrame to transform
            mapping_df: DataFrame containing column name mappings
            
        Returns:
            tuple: (Transformed DataFrame, transformation report)
        """
        if df is None or mapping_df is None:
            return df, {"success": False, "message": "Invalid input for transformation"}
        
        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Create mapping dictionary from mapping DataFrame
        mapping_dict = dict(zip(mapping_df['old_column_name'], mapping_df['new_column_name']))
        
        # Track applied and skipped transformations
        applied_mappings = []
        skipped_mappings = []
        
        # Check which columns in the mapping actually exist in the DataFrame
        df_columns = set(df.columns)
        
        for old_col, new_col in mapping_dict.items():
            if old_col in df_columns:
                # Apply the transformation
                result_df = result_df.rename(columns={old_col: new_col})
                applied_mappings.append((old_col, new_col))
            else:
                skipped_mappings.append((old_col, new_col))
        
        # Generate transformation report
        transformation_report = {
            "success": True,
            "original_columns": list(df.columns),
            "transformed_columns": list(result_df.columns),
            "applied_mappings": applied_mappings,
            "skipped_mappings": skipped_mappings,
            "total_columns": len(df.columns),
            "transformed_count": len(applied_mappings)
        }
        
        return result_df, transformation_report
    
    @staticmethod
    def generate_transformation_report(report: Dict[str, Any]) -> str:
        """
        Generate a human-readable report of the column transformations.
        
        Args:
            report: Dictionary containing transformation details
            
        Returns:
            str: Formatted transformation report
        """
        if not report.get("success", False):
            return f"Transformation failed: {report.get('message', 'Unknown error')}"
        
        # Build a formatted report
        report_text = """
        # Column Transformation Report

        ## Summary
        - Total columns in DataFrame: {total_columns}
        - Columns transformed: {transformed_count}
        - Columns unchanged: {unchanged_count}
        
        ## Applied Transformations
        """.format(
            total_columns=report["total_columns"],
            transformed_count=report["transformed_count"],
            unchanged_count=report["total_columns"] - report["transformed_count"]
        )
        
        # Add details of applied mappings
        if report["applied_mappings"]:
            report_text += "| Original Column | New Column |\n|----------------|------------|\n"
            for old_col, new_col in report["applied_mappings"]:
                report_text += f"| {old_col} | {new_col} |\n"
        else:
            report_text += "No column transformations were applied.\n"
        
        # Add skipped mappings if any
        if report["skipped_mappings"]:
            report_text += """
            ## Skipped Transformations
            The following mappings were skipped because the columns don't exist in the DataFrame:
            
            | Original Column | New Column |\n|----------------|------------|\n
            """
            for old_col, new_col in report["skipped_mappings"]:
                report_text += f"| {old_col} | {new_col} |\n"
        
        return report_text
    
    @staticmethod
    def validate_mapping(mapping_df: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the column mapping against the DataFrame.
        
        Args:
            mapping_df: DataFrame containing column mappings
            df: The DataFrame to transform
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        if mapping_df.empty:
            validation_results["valid"] = False
            validation_results["errors"].append("Mapping file is empty")
            return validation_results
        
        # Check for duplicate target column names
        duplicate_new_cols = mapping_df['new_column_name'].duplicated()
        if duplicate_new_cols.any():
            duplicates = mapping_df.loc[duplicate_new_cols, 'new_column_name'].unique().tolist()
            validation_results["warnings"].append(
                f"Duplicate target column names detected: {', '.join(duplicates)}"
            )
        
        # Check if all source columns exist in the DataFrame
        df_columns = set(df.columns)
        for old_col in mapping_df['old_column_name']:
            if old_col not in df_columns:
                validation_results["warnings"].append(
                    f"Source column '{old_col}' does not exist in the DataFrame"
                )
        
        # Check if any existing column will be overwritten
        existing_target_cols = set(mapping_df['new_column_name']) & df_columns
        non_mapped_existing = existing_target_cols - set(mapping_df['old_column_name'])
        if non_mapped_existing:
            validation_results["warnings"].append(
                f"These target columns already exist and may cause conflicts: {', '.join(non_mapped_existing)}"
            )
        
        return validation_results