import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
import re
from fuzzywuzzy import fuzz, process
import Levenshtein
from collections import defaultdict
import os
import datetime
try:
    from metaphone import doublemetaphone
    HAS_METAPHONE = True
except ImportError:
    HAS_METAPHONE = False
    print("Warning: metaphone package not installed. Phonetic matching will be disabled.")


class DuplicateDetector:
    """
    Handles detection of duplicate records in datasets, particularly focusing
    on name-based duplicate detection with fuzzy matching capabilities.
    
    This class provides methods to identify potential duplicate records using
    string similarity algorithms, generate detailed reports about duplicates,
    and offer suggestions for record consolidation.
    """
    
    def __init__(self):
        """Initialize the DuplicateDetector with default settings."""
        # Default thresholds for different similarity methods
        self.similarity_thresholds = {
            'exact': 100,           # Exact match
            'high': 85,             # Very similar
            'medium': 70,           # Moderately similar
            'low': 60               # Somewhat similar
        }
        
        # Additional thresholds for improved matching
        self.first_name_threshold_bonus = 10  # Bonus for first name threshold (more strict)
        self.max_levenshtein_distance = 2     # Maximum allowed character difference
        self.min_length_ratio = 0.7           # Minimum ratio of shorter name to longer name
        
    def detect_name_duplicates(
        self, 
        df: pd.DataFrame,
        first_name_col: str,
        last_name_col: str,
        threshold: str = 'medium',
        additional_cols: List[str] = None,
        use_transitive_closure: bool = True,
        use_advanced_matching: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect potential duplicate records based on first and last name similarity.
        
        Args:
            df (DataFrame): DataFrame containing the records to check
            first_name_col (str): Column name for first names
            last_name_col (str): Column name for last names
            threshold (str): Similarity threshold level ('exact', 'high', 'medium', 'low')
            additional_cols (list): Additional columns to include in the output
            use_transitive_closure (bool): Whether to use transitive closure for grouping
            use_advanced_matching (bool): Whether to use advanced matching techniques
            
        Returns:
            tuple: (DataFrame with duplicate groups, summary dictionary)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if first_name_col not in df.columns or last_name_col not in df.columns:
            raise ValueError(f"Columns {first_name_col} and/or {last_name_col} not found in DataFrame")
        
        # Get actual threshold value
        threshold_value = self.similarity_thresholds.get(threshold, self.similarity_thresholds['medium'])
        
        # Copy the input DataFrame to avoid modifying the original
        working_df = df.copy()
        
        # Create a full name column
        working_df['full_name'] = working_df.apply(
            lambda row: self._combine_names(
                str(row[first_name_col]).strip() if pd.notna(row[first_name_col]) else "",
                str(row[last_name_col]).strip() if pd.notna(row[last_name_col]) else ""
            ),
            axis=1
        )
        
        # Clean and standardize names
        working_df['clean_name'] = working_df['full_name'].apply(self._clean_and_standardize_name)
        working_df['clean_first_name'] = working_df[first_name_col].apply(
            lambda x: self._clean_and_standardize_name(str(x)) if pd.notna(x) else ""
        )
        working_df['clean_last_name'] = working_df[last_name_col].apply(
            lambda x: self._clean_and_standardize_name(str(x)) if pd.notna(x) else ""
        )
        
        # If using advanced matching techniques
        if use_advanced_matching:
            # First, group exact matches 
            exact_duplicates = working_df[working_df.duplicated(subset=['clean_name'], keep=False)].copy()
            exact_duplicates['duplicate_group'] = exact_duplicates.groupby('clean_name').ngroup() + 1
            exact_duplicates['match_type'] = 'Exact'
            exact_duplicates['similarity_score'] = 100.0
            
            # Next, find fuzzy matches among remaining records using improved methods
            remaining_df = working_df.drop(exact_duplicates.index) if not exact_duplicates.empty else working_df.copy()
            
            # Apply two-tier matching approach
            fuzzy_duplicates = self._find_fuzzy_duplicates_improved(
                remaining_df, 
                'clean_first_name',
                'clean_last_name',
                threshold_value,
                next_group_id=exact_duplicates['duplicate_group'].max() + 1 if not exact_duplicates.empty else 1
            )
        else:
            # Using original matching logic
            # First, group exact matches 
            exact_duplicates = working_df[working_df.duplicated(subset=['clean_name'], keep=False)].copy()
            exact_duplicates['duplicate_group'] = exact_duplicates.groupby('clean_name').ngroup() + 1
            exact_duplicates['match_type'] = 'Exact'
            exact_duplicates['similarity_score'] = 100.0
            
            # Next, find fuzzy matches among remaining records
            remaining_df = working_df.drop(exact_duplicates.index) if not exact_duplicates.empty else working_df.copy()
            
            # Identify fuzzy duplicates using the original approach
            fuzzy_duplicates = self._find_fuzzy_duplicates(
                remaining_df, 
                'clean_name',
                threshold_value,
                next_group_id=exact_duplicates['duplicate_group'].max() + 1 if not exact_duplicates.empty else 1
            )
        
        # Combine exact and fuzzy duplicates
        all_duplicates = pd.concat([exact_duplicates, fuzzy_duplicates]) if not fuzzy_duplicates.empty else exact_duplicates
        
        # If no duplicates found, return empty result
        if all_duplicates.empty:
            return pd.DataFrame(), {
                'total_records': len(df),
                'duplicate_groups': 0,
                'total_duplicates': 0,
                'percent_duplicates': 0.0,
                'threshold_used': threshold,
                'threshold_value': threshold_value
            }
        
        # Select columns for final output
        output_columns = ['duplicate_group', 'match_type', 'similarity_score', 
                         first_name_col, last_name_col, 'full_name']
        
        # Add additional columns if specified
        if additional_cols:
            valid_additional_cols = [col for col in additional_cols if col in df.columns]
            output_columns.extend(valid_additional_cols)
        
        # Prepare final result
        result_df = all_duplicates[output_columns].sort_values(
            by=['duplicate_group', 'similarity_score'], 
            ascending=[True, False]
        )
        
        # Generate summary information
        summary = {
            'total_records': len(df),
            'duplicate_groups': result_df['duplicate_group'].nunique(),
            'total_duplicates': len(result_df),
            'percent_duplicates': round(len(result_df) / len(df) * 100, 2),
            'threshold_used': threshold,
            'threshold_value': threshold_value,
            'duplicate_counts_by_type': result_df['match_type'].value_counts().to_dict()
        }
        
        return result_df, summary
    
    def _combine_names(self, first_name: str, last_name: str) -> str:
        """
        Combine first and last names into a full name.
        
        Args:
            first_name (str): First name 
            last_name (str): Last name
            
        Returns:
            str: Combined full name
        """
        return f"{first_name} {last_name}".strip()
    
    def _clean_and_standardize_name(self, name: str) -> str:
        """
        Clean and standardize a name for comparison.
        
        Args:
            name (str): Name to clean and standardize
            
        Returns:
            str: Cleaned and standardized name
        """
        if not name or not isinstance(name, str):
            return ""
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove common titles, suffixes, prefixes
        titles_and_suffixes = [
            'mr', 'mrs', 'miss', 'ms', 'dr', 'prof', 'rev', 'jr', 'sr', 'i', 'ii', 'iii', 'iv', 'v',
            'phd', 'md', 'dds', 'esq', 'hon'
        ]
        
        for item in titles_and_suffixes:
            name = re.sub(rf'\b{item}\b\.?', '', name)
        
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _find_fuzzy_duplicates(
        self, 
        df: pd.DataFrame, 
        name_col: str, 
        threshold: int,
        next_group_id: int = 1
    ) -> pd.DataFrame:
        """
        Find fuzzy duplicate records in the DataFrame (original implementation).
        
        Args:
            df (DataFrame): DataFrame to search for duplicates
            name_col (str): Column name containing standardized full names
            threshold (int): Similarity threshold score (0-100)
            next_group_id (int): Starting ID for duplicate groups
            
        Returns:
            DataFrame: DataFrame containing identified fuzzy duplicates
        """
        if df.empty:
            return pd.DataFrame()
        
        # Initialize structures to track duplicates
        duplicate_groups = defaultdict(list)
        processed_indices = set()
        group_id = next_group_id
        
        # Get all names as a list for comparison
        names_list = df[name_col].tolist()
        index_list = df.index.tolist()
        # Handle duplicates in names by keeping a list of indices for each name
        name_to_indices = {}
        for name, idx in zip(names_list, index_list):
            if name not in name_to_indices:
                name_to_indices[name] = []
            name_to_indices[name].append(idx)
        
        # For each record, find potential fuzzy matches
        for i, (idx, name) in enumerate(zip(index_list, names_list)):
            if idx in processed_indices:
                continue
            
            # Check against remaining names
            remaining_indices = index_list[i+1:]
            remaining_names = names_list[i+1:]
            
            if not remaining_names:
                continue
            
            # Use fuzzywuzzy to find matches
            matches = process.extractBests(
                name, 
                remaining_names, 
                scorer=fuzz.token_sort_ratio, 
                score_cutoff=threshold,
                limit=100  # Limit to avoid excessive matches
            )
            
            if matches:
                # If matches found, create a new duplicate group
                duplicate_groups[group_id].append({
                    'index': idx,
                    'name': name,
                    'match_type': 'Primary',
                    'similarity_score': 100.0
                })
                processed_indices.add(idx)
                
                # Add all matches to the same group
                for match_name, score in matches:
                    for match_idx in name_to_indices.get(match_name, []):
                        if match_idx not in processed_indices:
                            duplicate_groups[group_id].append({
                                'index': match_idx,
                                'name': match_name,
                                'match_type': 'Fuzzy',
                                'similarity_score': float(score)
                            })
                            processed_indices.add(match_idx)
                
                group_id += 1
        
        # Convert the duplicate groups to a DataFrame
        duplicate_records = []
        for group_id, records in duplicate_groups.items():
            for record in records:
                duplicate_records.append({
                    'index': record['index'],
                    'duplicate_group': group_id,
                    'match_type': record['match_type'],
                    'similarity_score': record['similarity_score']
                })
        
        if not duplicate_records:
            return pd.DataFrame()
        
        # Create DataFrame from duplicate records and join with original data
        duplicates_df = pd.DataFrame(duplicate_records).set_index('index')
        result = df.join(duplicates_df)
        
        return result[result['duplicate_group'].notna()].copy()

    def _find_fuzzy_duplicates_improved(
        self, 
        df: pd.DataFrame, 
        first_name_col: str,
        last_name_col: str,
        threshold: int,
        next_group_id: int = 1
    ) -> pd.DataFrame:
        """
        Find fuzzy duplicate records in the DataFrame using improved matching techniques.
        
        Args:
            df (DataFrame): DataFrame to search for duplicates
            first_name_col (str): Column name for first names
            last_name_col (str): Column name for last names
            threshold (int): Similarity threshold score (0-100)
            next_group_id (int): Starting ID for duplicate groups
            
        Returns:
            DataFrame: DataFrame containing identified fuzzy duplicates
        """
        if df.empty:
            return pd.DataFrame()
        
        # Initialize structures to track duplicates
        duplicate_groups = defaultdict(list)
        processed_indices = set()
        group_id = next_group_id
        
        # First, group by exact last name matches to reduce comparison scope
        last_name_groups = df.groupby(last_name_col)
        
        # Adjust thresholds
        first_name_threshold = threshold + self.first_name_threshold_bonus  # Higher threshold for first names
        last_name_threshold = threshold
        
        # Process each last name group
        for last_name, group in last_name_groups:
            if len(group) <= 1:
                continue  # Skip groups with only one record
            
            # For each record in the group, compare with other records in the same group
            records = group.to_dict('records')
            indices = group.index.tolist()
            
            for i, (record, idx) in enumerate(zip(records, indices)):
                if idx in processed_indices:
                    continue
                
                first_name = record[first_name_col]
                
                # Check against remaining records in the same last name group
                for j in range(i + 1, len(records)):
                    compare_record = records[j]
                    compare_idx = indices[j]
                    
                    if compare_idx in processed_indices:
                        continue
                    
                    compare_first_name = compare_record[first_name_col]
                    
                    # Apply improved matching logic
                    is_match, match_score = self._is_name_match(
                        first_name, compare_first_name, 
                        last_name, last_name,  # Last names are already same in this group
                        first_name_threshold, last_name_threshold
                    )
                    
                    if is_match:
                        # If this is the first match for this record, create a new group
                        if idx not in processed_indices:
                            duplicate_groups[group_id].append({
                                'index': idx,
                                'first_name': first_name,
                                'last_name': last_name,
                                'match_type': 'Primary',
                                'similarity_score': 100.0
                            })
                            processed_indices.add(idx)
                        
                        duplicate_groups[group_id].append({
                            'index': compare_idx,
                            'first_name': compare_first_name,
                            'last_name': last_name,
                            'match_type': 'Fuzzy',
                            'similarity_score': float(match_score)
                        })
                        processed_indices.add(compare_idx)
                
                # If we added matches to this group, increment the group ID
                if group_id in duplicate_groups and len(duplicate_groups[group_id]) > 0:
                    group_id += 1
        
        # Also check for similar last names with same/similar first names
        # Get all unique combinations
        all_records = df.to_dict('records')
        all_indices = df.index.tolist()
        
        for i, (record, idx) in enumerate(zip(all_records, all_indices)):
            if idx in processed_indices:
                continue
            
            first_name = record[first_name_col]
            last_name = record[last_name_col]
            
            # Check against all other records
            for j in range(i + 1, len(all_records)):
                compare_record = all_records[j]
                compare_idx = all_indices[j]
                
                if compare_idx in processed_indices:
                    continue
                
                compare_first_name = compare_record[first_name_col]
                compare_last_name = compare_record[last_name_col]
                
                # Skip if last names are identical (already handled above)
                if last_name == compare_last_name:
                    continue
                
                # Apply improved matching logic for cross-group matching
                # Here we need high similarity on both first and last name
                is_match, match_score = self._is_name_match(
                    first_name, compare_first_name,
                    last_name, compare_last_name,
                    threshold, threshold, 
                    require_both=True  # Require both first and last name to be similar
                )
                
                if is_match:
                    # If this is the first match for this record, create a new group
                    if idx not in processed_indices:
                        duplicate_groups[group_id].append({
                            'index': idx,
                            'first_name': first_name,
                            'last_name': last_name,
                            'match_type': 'Primary',
                            'similarity_score': 100.0
                        })
                        processed_indices.add(idx)
                    
                    duplicate_groups[group_id].append({
                        'index': compare_idx,
                        'first_name': compare_first_name,
                        'last_name': compare_last_name,
                        'match_type': 'Fuzzy',
                        'similarity_score': float(match_score)
                    })
                    processed_indices.add(compare_idx)
                    
                    # Increment group_id only if we added matches
                    if len(duplicate_groups[group_id]) > 0:
                        group_id += 1
        
        # Convert the duplicate groups to a DataFrame
        duplicate_records = []
        for group_id, records in duplicate_groups.items():
            if len(records) > 1:  # Only include groups with more than one record
                for record in records:
                    duplicate_records.append({
                        'index': record['index'],
                        'duplicate_group': group_id,
                        'match_type': record['match_type'],
                        'similarity_score': record['similarity_score']
                    })
        
        if not duplicate_records:
            return pd.DataFrame()
        
        # Create DataFrame from duplicate records and join with original data
        duplicates_df = pd.DataFrame(duplicate_records).set_index('index')
        result = df.join(duplicates_df)
        
        return result[result['duplicate_group'].notna()].copy()
    
    def _is_name_match(
        self,
        first_name1: str, 
        first_name2: str,
        last_name1: str,
        last_name2: str,
        first_name_threshold: int,
        last_name_threshold: int,
        require_both: bool = False
    ) -> Tuple[bool, int]:
        """
        Check if two name pairs are a potential match using multiple criteria.
        
        Args:
            first_name1 (str): First name from first record
            first_name2 (str): First name from second record
            last_name1 (str): Last name from first record
            last_name2 (str): Last name from second record
            first_name_threshold (int): Similarity threshold for first names
            last_name_threshold (int): Similarity threshold for last names
            require_both (bool): Whether to require both first and last name to pass threshold
            
        Returns:
            tuple: (is_match, similarity_score)
        """
        # Clean input
        fn1 = self._clean_and_standardize_name(str(first_name1)) if first_name1 else ""
        fn2 = self._clean_and_standardize_name(str(first_name2)) if first_name2 else ""
        ln1 = self._clean_and_standardize_name(str(last_name1)) if last_name1 else ""
        ln2 = self._clean_and_standardize_name(str(last_name2)) if last_name2 else ""
        
        # Skip empty names
        if not fn1 or not fn2 or not ln1 or not ln2:
            return False, 0
        
        # Check length ratio
        if not self._check_length_ratio(fn1, fn2):
            return False, 0
        
        # Check Levenshtein distance for first names
        if not self._check_levenshtein_distance(fn1, fn2):
            return False, 0
        
        # Calculate similarity scores
        first_name_score = fuzz.token_sort_ratio(fn1, fn2)
        last_name_score = fuzz.token_sort_ratio(ln1, ln2)
        
        # Check phonetic similarity if available
        if HAS_METAPHONE and not self._compare_names_phonetically(fn1, fn2):
            # Phonetic check failed, reduce score
            first_name_score = max(0, first_name_score - 20)
        
        # Calculate overall score
        overall_score = (first_name_score + last_name_score) / 2
        
        # Determine if it's a match based on thresholds
        if require_both:
            is_match = (first_name_score >= first_name_threshold and 
                       last_name_score >= last_name_threshold)
        else:
            is_match = (first_name_score >= first_name_threshold or 
                       last_name_score >= last_name_threshold)
        
        return is_match, overall_score
    
    def _check_length_ratio(self, name1: str, name2: str) -> bool:
        """
        Check if the length ratio between two names is acceptable.
        
        Args:
            name1 (str): First name
            name2 (str): Second name
            
        Returns:
            bool: True if the length ratio is acceptable
        """
        len1, len2 = len(name1), len(name2)
        if len1 == 0 or len2 == 0:
            return False
        
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        
        # If one name is less than the minimum ratio of the other, not a match
        return min_len / max_len >= self.min_length_ratio
    
    def _check_levenshtein_distance(self, name1: str, name2: str) -> bool:
        """
        Check if the Levenshtein distance between two names is acceptable.
        
        Args:
            name1 (str): First name
            name2 (str): Second name
            
        Returns:
            bool: True if the Levenshtein distance is acceptable
        """
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(name1, name2)
        
        # If the difference is more than the maximum allowed distance, not a match
        return distance <= self.max_levenshtein_distance
    
    def _compare_names_phonetically(self, name1: str, name2: str) -> bool:
        """
        Compare two names using phonetic algorithms.
        
        Args:
            name1 (str): First name
            name2 (str): Second name
            
        Returns:
            bool: True if the names are phonetically similar
        """
        if not HAS_METAPHONE:
            return True  # Skip this check if metaphone isn't available
        
        # Get the phonetic codes for both names
        code1 = doublemetaphone(name1)
        code2 = doublemetaphone(name2)
        
        # Compare the phonetic codes
        # Only consider a match if at least one phonetic representation is similar
        return code1[0] == code2[0] or (code1[1] and code2[1] and code1[1] == code2[1])
    
    def generate_duplicate_report(
        self, 
        duplicate_df: pd.DataFrame, 
        summary: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive report of duplicate findings.
        
        Args:
            duplicate_df (DataFrame): DataFrame containing identified duplicates
            summary (dict): Summary information about the duplicate detection
            
        Returns:
            str: Formatted report as markdown text
        """
        if duplicate_df.empty:
            return """
            # Duplicate Detection Report
            
            No duplicate records were found in the dataset.
            
            ## Analysis Parameters
            - Threshold used: {threshold_used} ({threshold_value})
            - Total records analyzed: {total_records}
            """.format(**summary)
        
        # Build the report
        report = """
        # Duplicate Detection Report
        
        ## Summary
        - Total records analyzed: {total_records}
        - Duplicate groups found: {duplicate_groups}
        - Total potential duplicates: {total_duplicates}
        - Percentage of duplicates: {percent_duplicates}%
        - Similarity threshold: {threshold_used} ({threshold_value})
        
        ## Duplicate Groups
        """.format(**summary)
        
        # Add information about each duplicate group
        for group_id in sorted(duplicate_df['duplicate_group'].unique()):
            group_df = duplicate_df[duplicate_df['duplicate_group'] == group_id]
            
            report += f"""
        ### Group {int(group_id)}
        - Number of records: {len(group_df)}
        - Match types: {', '.join(group_df['match_type'].unique())}
            
        | Match Type | Similarity | Name |
        |------------|------------|------|
        """
            
            for _, row in group_df.iterrows():
                report += f"| {row['match_type']} | {row['similarity_score']:.1f} | {row['full_name']} |\n"
            
            report += "\n"
        
        # Recommendations
        report += """
        ## Recommendations
        - Review each duplicate group to confirm if they are true duplicates
        - Consider merging records with similarity scores above 90
        - Manual verification recommended for records with similarity between 70-90
        """
        
        return report
    
    def export_duplicates_to_csv(
        self,
        duplicate_df: pd.DataFrame,
        custom_path: str = None
    ) -> str:
        """
        Export duplicate records to a CSV file.
        
        Args:
            duplicate_df (DataFrame): DataFrame containing identified duplicates
            custom_path (str, optional): Custom path to save the CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        if duplicate_df.empty:
            return None
        
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set the default path if not provided
        if not custom_path:
            filepath = f"/Users/omkar/Downloads/duplicate_records_{timestamp}.csv"
        else:
            # If custom path is a directory, add the filename
            if os.path.isdir(custom_path):
                filepath = os.path.join(custom_path, f"duplicate_records_{timestamp}.csv")
            else:
                filepath = custom_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the DataFrame to CSV
        duplicate_df.to_csv(filepath, index=False)
        
        return filepath
    
    def find_potential_matches(
        self, 
        df: pd.DataFrame,
        query_name: str,
        first_name_col: str = None,
        last_name_col: str = None,
        name_col: str = None,
        threshold: str = 'medium',
        max_matches: int = 10
    ) -> pd.DataFrame:
        """
        Find potential matches for a query name in the DataFrame.
        
        Args:
            df (DataFrame): DataFrame to search for matches
            query_name (str): Name to find matches for
            first_name_col (str, optional): Column name for first names
            last_name_col (str, optional): Column name for last names
            name_col (str, optional): Column name for full names
            threshold (str): Similarity threshold level
            max_matches (int): Maximum number of matches to return
            
        Returns:
            DataFrame: DataFrame containing potential matches
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Get actual threshold value
        threshold_value = self.similarity_thresholds.get(threshold, self.similarity_thresholds['medium'])
        
        # Prepare the query name
        clean_query = self._clean_and_standardize_name(query_name)
        
        # Prepare the DataFrame for comparison
        working_df = df.copy()
        
        if name_col and name_col in df.columns:
            # Use the provided full name column
            working_df['clean_name'] = working_df[name_col].apply(self._clean_and_standardize_name)
        elif first_name_col and last_name_col and first_name_col in df.columns and last_name_col in df.columns:
            # Combine first and last names
            working_df['full_name'] = working_df.apply(
                lambda row: self._combine_names(
                    str(row[first_name_col]).strip() if pd.notna(row[first_name_col]) else "",
                    str(row[last_name_col]).strip() if pd.notna(row[last_name_col]) else ""
                ),
                axis=1
            )
            working_df['clean_name'] = working_df['full_name'].apply(self._clean_and_standardize_name)
        else:
            raise ValueError("Either name_col or both first_name_col and last_name_col must be provided")
        
        # Find matches using improved methods
        if first_name_col and last_name_col:
            # Split query into first and last name
            parts = clean_query.split()
            if len(parts) >= 2:
                query_first = parts[0]
                query_last = parts[-1]
                
                # Add clean first/last name columns
                working_df['clean_first_name'] = working_df[first_name_col].apply(
                    lambda x: self._clean_and_standardize_name(str(x)) if pd.notna(x) else ""
                )
                working_df['clean_last_name'] = working_df[last_name_col].apply(
                    lambda x: self._clean_and_standardize_name(str(x)) if pd.notna(x) else ""
                )
                
                # Calculate similarity scores
                working_df['first_name_similarity'] = working_df['clean_first_name'].apply(
                    lambda x: fuzz.token_sort_ratio(query_first, x)
                )
                working_df['last_name_similarity'] = working_df['clean_last_name'].apply(
                    lambda x: fuzz.token_sort_ratio(query_last, x)
                )
                
                # Apply length and Levenshtein filters
                working_df['length_ratio_ok'] = working_df['clean_first_name'].apply(
                    lambda x: self._check_length_ratio(query_first, x)
                )
                working_df['levenshtein_ok'] = working_df['clean_first_name'].apply(
                    lambda x: self._check_levenshtein_distance(query_first, x)
                )
                
                # Calculate overall similarity
                working_df['similarity_score'] = (
                    working_df['first_name_similarity'] * 0.6 + 
                    working_df['last_name_similarity'] * 0.4
                )
                
                # Apply all filters
                matches = working_df[
                    (working_df['similarity_score'] >= threshold_value) &
                    (working_df['length_ratio_ok']) &
                    (working_df['levenshtein_ok'])
                ].sort_values(by='similarity_score', ascending=False).head(max_matches)
            else:
                # Fall back to standard matching for short names
                working_df['similarity_score'] = working_df['clean_name'].apply(
                    lambda x: fuzz.token_sort_ratio(clean_query, x)
                )
                matches = working_df[working_df['similarity_score'] >= threshold_value].sort_values(
                    by='similarity_score', ascending=False
                ).head(max_matches)
        else:
            # Fall back to standard matching if columns aren't specified
            working_df['similarity_score'] = working_df['clean_name'].apply(
                lambda x: fuzz.token_sort_ratio(clean_query, x)
            )
            matches = working_df[working_df['similarity_score'] >= threshold_value].sort_values(
                by='similarity_score', ascending=False
            ).head(max_matches)
        
        if matches.empty:
            return pd.DataFrame()
        
        # Add match type classification
        matches['match_type'] = matches['similarity_score'].apply(
            lambda score: 
                'Exact' if score == 100 else
                'High' if score >= 90 else
                'Medium' if score >= 80 else
                'Low'
        )
        
        return matches
    
    def set_similarity_threshold(self, threshold_name: str, value: int) -> None:
        """
        Set a custom similarity threshold.
        
        Args:
            threshold_name (str): Name of the threshold ('exact', 'high', 'medium', 'low')
            value (int): Threshold value (0-100)
        """
        if threshold_name not in self.similarity_thresholds:
            raise ValueError(f"Invalid threshold name: {threshold_name}")
        
        if not 0 <= value <= 100:
            raise ValueError("Threshold value must be between 0 and 100")
        
        self.similarity_thresholds[threshold_name] = value
    
    def set_advanced_parameters(
        self,
        first_name_threshold_bonus: int = None,
        max_levenshtein_distance: int = None,
        min_length_ratio: float = None
    ) -> None:
        """
        Set advanced parameters for duplicate detection.
        
        Args:
            first_name_threshold_bonus (int): Bonus threshold for first names
            max_levenshtein_distance (int): Maximum allowed character difference
            min_length_ratio (float): Minimum ratio of shorter name to longer name
        """
        if first_name_threshold_bonus is not None:
            self.first_name_threshold_bonus = first_name_threshold_bonus
        
        if max_levenshtein_distance is not None:
            self.max_levenshtein_distance = max_levenshtein_distance
        
        if min_length_ratio is not None:
            if not 0 <= min_length_ratio <= 1:
                raise ValueError("min_length_ratio must be between 0 and 1")
            self.min_length_ratio = min_length_ratio
        
    def get_similarity_score(self, name1: str, name2: str, method: str = 'token_sort') -> int:
        """
        Calculate similarity score between two names.
        
        Args:
            name1 (str): First name
            name2 (str): Second name
            method (str): Similarity method ('token_sort', 'token_set', 'partial', 'levenshtein')
            
        Returns:
            int: Similarity score (0-100)
        """
        clean_name1 = self._clean_and_standardize_name(name1)
        clean_name2 = self._clean_and_standardize_name(name2)
        
        if method == 'token_sort':
            return fuzz.token_sort_ratio(clean_name1, clean_name2)
        elif method == 'token_set':
            return fuzz.token_set_ratio(clean_name1, clean_name2)
        elif method == 'partial':
            return fuzz.partial_ratio(clean_name1, clean_name2)
        elif method == 'levenshtein':
            # Levenshtein distance converted to a similarity score
            max_len = max(len(clean_name1), len(clean_name2))
            if max_len == 0:
                return 100  # Both strings are empty
            distance = Levenshtein.distance(clean_name1, clean_name2)
            return max(0, int(100 * (1 - distance / max_len)))
        else:
            raise ValueError(f"Unsupported similarity method: {method}")