"""
Test file for Data Processor
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test the Data Processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        
        # Create test data
        self.test_data = {
            'Citi': pd.DataFrame({
                'date': ['2024-01-01', '2024-01-15', '2024-02-01'],
                'description': ['Amazon Purchase', 'Starbucks Coffee', 'Uber Ride'],
                'amount': ['$100.50', '$5.25', '$25.00'],
                'category': ['Shopping', 'Food & Drink', 'Transport']
            }),
            'Chase': pd.DataFrame({
                'date': ['2024-01-05', '2024-01-20', '2024-02-05'],
                'description': ['Netflix Subscription', 'Whole Foods', 'Shell Gas'],
                'amount': ['$15.99', '$75.30', '$45.00'],
                'category': ['Entertainment', 'Food', 'Fuel']
            }),
            'Cash': pd.DataFrame({
                'date': ['2024-01-10', '2024-01-25', '2024-02-10'],
                'description': ['Target Shopping', 'Restaurant', 'Movie Tickets'],
                'amount': ['$85.00', '$120.50', '$35.00'],
                'category': ['Retail', 'Restaurants', 'Movies']
            })
        }
    
    def test_combine_tabs_data(self):
        """Test combining data from multiple tabs"""
        combined_df = self.processor.combine_tabs_data(self.test_data)
        
        self.assertIsInstance(combined_df, pd.DataFrame)
        self.assertEqual(len(combined_df), 9)  # 3 rows per tab * 3 tabs
        self.assertIn('tab_name', combined_df.columns)
        
        # Check that all tab names are present
        tab_names = combined_df['tab_name'].unique()
        expected_tabs = ['Citi', 'Chase', 'Cash']
        for tab in expected_tabs:
            self.assertIn(tab, tab_names)
    
    def test_combine_tabs_data_empty(self):
        """Test combining data with empty tabs"""
        empty_data = {
            'Citi': pd.DataFrame(),
            'Chase': pd.DataFrame(),
            'Cash': pd.DataFrame()
        }
        
        with self.assertRaises(Exception):
            self.processor.combine_tabs_data(empty_data)
    
    def test_clean_column_names(self):
        """Test cleaning column names"""
        # Create test dataframe with messy column names
        test_df = pd.DataFrame({
            'Transaction_Date': ['2024-01-01'],
            'Merchant': ['Test'],
            'Transaction_Amount': [100.0],
            'Transaction_Category': ['Food'],
            '  Extra_Whitespace  ': ['Extra']
        })
        
        cleaned_df = self.processor.clean_column_names(test_df)
        
        # Check that column names are cleaned
        self.assertIn('date', cleaned_df.columns)
        self.assertIn('description', cleaned_df.columns)
        self.assertIn('amount', cleaned_df.columns)
        self.assertIn('category', cleaned_df.columns)
        self.assertIn('extra_whitespace', cleaned_df.columns)
    
    def test_validate_required_columns_valid(self):
        """Test validation with all required columns present"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01'],
            'description': ['Test'],
            'amount': [100.0],
            'category': ['Food']
        })
        
        required_columns = ['date', 'description', 'amount', 'category']
        result = self.processor.validate_required_columns(test_df)
        
        self.assertTrue(result)
    
    def test_validate_required_columns_missing(self):
        """Test validation with missing required columns"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01'],
            'description': ['Test']
            # Missing 'amount' and 'category' columns
        })
        
        required_columns = ['date', 'description', 'amount', 'category']
        result = self.processor.validate_required_columns(test_df)
        
        self.assertFalse(result)
    
    def test_clean_date_column(self):
        """Test cleaning date column"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-02-15', '2024-03-30'],
            'description': ['Test1', 'Test2', 'Test3'],
            'amount': [100.0, 200.0, 300.0],
            'category': ['Food', 'Food', 'Food']
        })
        
        cleaned_df = self.processor.clean_date_column(test_df)
        
        # Check that dates are converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['date']))
        self.assertEqual(len(cleaned_df), 3)  # All dates should be valid
    
    def test_clean_date_column_invalid_dates(self):
        """Test cleaning date column with invalid dates"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01', 'invalid_date', '2024-03-30'],
            'description': ['Test1', 'Test2', 'Test3'],
            'amount': [100.0, 200.0, 300.0],
            'category': ['Food', 'Food', 'Food']
        })
        
        cleaned_df = self.processor.clean_date_column(test_df)
        
        # Check that invalid dates are removed
        self.assertEqual(len(cleaned_df), 2)  # Only 2 valid dates
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['date']))
    
    def test_clean_amount_column(self):
        """Test cleaning amount column"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'description': ['Test1', 'Test2', 'Test3'],
            'amount': ['$100.50', '200.75', 'invalid_amount'],
            'category': ['Food', 'Food', 'Food']
        })
        
        cleaned_df = self.processor.clean_amount_column(test_df)
        
        # Check that amounts are converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['amount']))
        self.assertEqual(len(cleaned_df), 2)  # Only 2 valid amounts
        self.assertEqual(cleaned_df['amount'].iloc[0], 100.50)
        self.assertEqual(cleaned_df['amount'].iloc[1], 200.75)
    
    def test_clean_category_column(self):
        """Test cleaning category column"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'description': ['Test1', 'Test2', 'Test3'],
            'amount': [100.0, 200.0, 300.0],
            'category': ['Food & Drink', 'Transport', 'Entertainment']
        })
        
        cleaned_df = self.processor.clean_category_column(test_df)
        
        # Check that categories are cleaned and mapped
        categories = cleaned_df['category'].unique()
        self.assertIn('Food', categories)  # 'Food & Drink' should be mapped to 'Food'
        self.assertIn('Transportation', categories)  # 'Transport' should be mapped to 'Transportation'
        self.assertIn('Entertainment', categories)
    
    def test_clean_description_column(self):
        """Test cleaning description column"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'description': ['AMZN MKTP US', 'STARBUCKS COFFEE', '  Target Shopping  '],
            'amount': [100.0, 200.0, 300.0],
            'category': ['Food', 'Food', 'Food']
        })
        
        cleaned_df = self.processor.clean_description_column(test_df)
        
        # Check that descriptions are cleaned
        descriptions = cleaned_df['description'].tolist()
        self.assertIn('Amazon', descriptions)  # 'AMZN MKTP US' should be mapped to 'Amazon'
        self.assertIn('Starbucks', descriptions)  # 'STARBUCKS COFFEE' should be mapped to 'Starbucks'
        self.assertIn('Target Shopping', descriptions)  # Whitespace should be trimmed
    
    def test_add_time_periods(self):
        """Test adding time period columns"""
        test_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-04-15', '2024-07-15', '2024-10-15']),
            'description': ['Test1', 'Test2', 'Test3', 'Test4'],
            'amount': [100.0, 200.0, 300.0, 400.0],
            'category': ['Food', 'Food', 'Food', 'Food']
        })
        
        processed_df = self.processor.add_time_periods(test_df)
        
        # Check that time period columns are added
        expected_columns = ['year', 'month', 'quarter', 'week', 'day_of_week', 'quarter_label', 'month_label']
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)
        
        # Check specific values
        self.assertEqual(processed_df['year'].iloc[0], 2024)
        self.assertEqual(processed_df['quarter'].iloc[0], 1)
        self.assertEqual(processed_df['quarter_label'].iloc[0], 'Q1')
        self.assertEqual(processed_df['month_label'].iloc[0], 'January')
    
    def test_remove_duplicates(self):
        """Test removing duplicate transactions"""
        test_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'description': ['Test1', 'Test1', 'Test2'],
            'amount': [100.0, 100.0, 200.0],
            'category': ['Food', 'Food', 'Food']
        })
        
        cleaned_df = self.processor.remove_duplicates(test_df)
        
        # Check that duplicates are removed
        self.assertEqual(len(cleaned_df), 2)  # Should have 2 unique rows
    
    def test_filter_valid_transactions(self):
        """Test filtering valid transactions"""
        test_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2025-01-01']),  # Future date
            'description': ['Test1', 'Test2', 'Test3'],
            'amount': [100.0, 0.01, -50.0],  # Negative amount
            'category': ['Food', 'Food', 'Food']
        })
        
        filtered_df = self.processor.filter_valid_transactions(test_df, min_amount=0.01)
        
        # Check that invalid transactions are filtered out
        self.assertEqual(len(filtered_df), 1)  # Only the valid transaction should remain
        self.assertEqual(filtered_df['amount'].iloc[0], 100.0)
    
    def test_process_transaction_data_complete_pipeline(self):
        """Test the complete data processing pipeline"""
        processed_df = self.processor.process_transaction_data(self.test_data)
        
        # Check that the pipeline completed successfully
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertFalse(processed_df.empty)
        
        # Check that all required columns are present
        required_columns = ['date', 'description', 'amount', 'category', 'tab_name']
        for col in required_columns:
            self.assertIn(col, processed_df.columns)
        
        # Check that time period columns are added
        time_columns = ['year', 'month', 'quarter', 'quarter_label', 'month_label']
        for col in time_columns:
            self.assertIn(col, processed_df.columns)
        
        # Check that data types are correct
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['amount']))
    
    def test_get_data_summary(self):
        """Test generating data summary"""
        # Create a processed dataframe
        processed_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'description': ['Test1', 'Test2', 'Test3'],
            'amount': [100.0, 200.0, 300.0],
            'category': ['Food', 'Food', 'Transportation'],
            'tab_name': ['Citi', 'Chase', 'Cash']
        })
        
        summary = self.processor.get_data_summary(processed_df)
        
        # Check summary structure
        self.assertIsInstance(summary, dict)
        self.assertIn('total_transactions', summary)
        self.assertIn('total_amount', summary)
        self.assertIn('categories', summary)
        self.assertIn('tabs', summary)
        
        # Check summary values
        self.assertEqual(summary['total_transactions'], 3)
        self.assertEqual(summary['total_amount'], 600.0)
        self.assertEqual(summary['categories']['Food'], 2)
        self.assertEqual(summary['categories']['Transportation'], 1)
    
    def test_get_data_summary_empty(self):
        """Test generating data summary for empty dataframe"""
        empty_df = pd.DataFrame()
        
        summary = self.processor.get_data_summary(empty_df)
        
        self.assertIn('error', summary)
        self.assertEqual(summary['error'], 'No data available')


if __name__ == '__main__':
    unittest.main() 