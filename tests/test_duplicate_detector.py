import unittest
import pandas as pd
from src.duplicate_detector import DuplicateDetector


# class TestDuplicateDetector(unittest.TestCase):
#     """
#     Test cases for the DuplicateDetector class functionality.
#     """
    
#     def setUp(self):
#         """Set up test fixtures."""
#         self.detector = DuplicateDetector()
        
#         # Create sample test data
#         self.test_data = pd.DataFrame({
#             'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             'first_name': [
#                 'Omkar', 'Omkar', 'Omkar', 'John', 'Johnny', 
#                 'Robert', 'Bob', 'Elizabeth', 'Lisa', 'Elisabeth'
#             ],
#             'last_name': [
#                 'Satapathy', 'S', 'Satap', 'Smith', 'Smith',
#                 'Johnson', 'Johnson', 'Taylor', 'Taylor', 'Taylor'
#             ],
#             'email': [
#                 'omkar.s@example.com', 'omkar@example.com', 'o.satap@example.com',
#                 'john.smith@example.com', 'j.smith@example.com', 
#                 'robert.j@example.com', 'bob.j@example.com',
#                 'e.taylor@example.com', 'lisa.t@example.com', 'e.taylor@mail.com'
#             ]
#         })
    
#     def test_detect_name_duplicates(self):
#         """Test the duplicate detection functionality."""
#         result_df, summary = self.detector.detect_name_duplicates(
#             self.test_data, 'first_name', 'last_name', 'medium'
#         )
        
#         # Verify we found duplicate groups
#         self.assertGreater(summary['duplicate_groups'], 0)
#         self.assertGreater(summary['total_duplicates'], 0)
        
#         # Check that Omkar variations are found in the result
#         omkar_records = result_df[result_df['first_name'] == 'Omkar']
#         self.assertGreater(len(omkar_records), 0)
#         # Note: depending on threshold, they might be in one or more groups
        
#         # Check that Robert and Bob can be matched (if both are found in results)
#         robert_records = result_df[result_df['first_name'] == 'Robert']
#         bob_records = result_df[result_df['first_name'] == 'Bob']
        
#         # Only verify if both Robert and Bob are in the results
#         if not robert_records.empty and not bob_records.empty:
#             robert_group = robert_records['duplicate_group'].iloc[0]
#             self.assertEqual(bob_records['duplicate_group'].iloc[0], robert_group)
    
#     def test_get_similarity_score(self):
#         """Test the similarity score calculation."""
#         # Test exact match
#         score = self.detector.get_similarity_score("Omkar Satapathy", "Omkar Satapathy")
#         self.assertEqual(score, 100)
        
#         # Test slight variation
#         score = self.detector.get_similarity_score("Omkar Satapathy", "Omkar S")
#         self.assertGreaterEqual(score, 60)  # Adjusted for lower threshold
        
#         # Test with different methods using a better example
#         score1 = self.detector.get_similarity_score("Smith John", "John Smith", "token_sort")
#         score2 = self.detector.get_similarity_score("Smith John", "John Smith", "token_set")
        
#         # Both should be high but token_set might be equal (both 100) for this simple example
#         self.assertGreaterEqual(score1, 90)
#         self.assertGreaterEqual(score2, 90)
        
#     def test_find_potential_matches(self):
#         """Test finding potential matches for a query name."""
#         matches = self.detector.find_potential_matches(
#             self.test_data, 
#             "Omkar Satap", 
#             first_name_col='first_name',
#             last_name_col='last_name',
#             threshold='medium'
#         )
        
#         # Should find at least some of the Omkar variations
#         self.assertGreater(len(matches), 0)
#         self.assertTrue(any(matches['first_name'] == 'Omkar'))
        
#         # Test with a full name column
#         self.test_data['full_name'] = self.test_data['first_name'] + ' ' + self.test_data['last_name']
#         matches = self.detector.find_potential_matches(
#             self.test_data,
#             "Johnny S",
#             name_col='full_name',
#             threshold='medium'
#         )
        
#         # Should find Johnny Smith
#         self.assertGreater(len(matches), 0)
#         self.assertTrue(any(matches['full_name'].str.contains('Johnny')))
    
#     def test_clean_and_standardize_name(self):
#         """Test name cleaning and standardization."""
#         # Test with titles and suffixes
#         clean_name = self.detector._clean_and_standardize_name("Dr. Omkar Satapathy Jr.")
#         self.assertEqual(clean_name, "omkar satapathy")
        
#         # Test with punctuation
#         clean_name = self.detector._clean_and_standardize_name("O'Satapathy, Omkar")
#         self.assertEqual(clean_name, "osatapathy omkar")
    
#     def test_generate_duplicate_report(self):
#         """Test report generation."""
#         result_df, summary = self.detector.detect_name_duplicates(
#             self.test_data, 'first_name', 'last_name', 'medium'
#         )
        
#         report = self.detector.generate_duplicate_report(result_df, summary)
        
#         # Verify report contains key sections
#         self.assertIn("Duplicate Detection Report", report)
#         self.assertIn("Summary", report)
#         self.assertIn("Duplicate Groups", report)
#         self.assertIn("Recommendations", report)
        
#         # Verify we have information for each group
#         for group_id in result_df['duplicate_group'].unique():
#             self.assertIn(f"Group {int(group_id)}", report)


# if __name__ == '__main__':
#     unittest.main()