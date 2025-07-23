"""
Test file for Pandas Executor
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pandas_executor import PandasExecutor, SafePandasExecutor


class TestPandasExecutor(unittest.TestCase):
    """Test the Pandas Executor"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test dataframe
        self.test_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15', '2024-03-01']),
            'description': ['Amazon Purchase', 'Starbucks Coffee', 'Uber Ride', 'Netflix Subscription', 'Whole Foods'],
            'amount': [100.50, 5.25, 25.00, 15.99, 75.30],
            'category': ['Shopping', 'Food', 'Transportation', 'Entertainment', 'Food'],
            'tab_name': ['Citi', 'Citi', 'Chase', 'Chase', 'Cash']
        })
        
        self.executor = PandasExecutor(self.test_data)
    
    def test_initialization(self):
        """Test executor initialization"""
        self.assertIsNotNone(self.executor.original_df)
        self.assertIsNotNone(self.executor.current_df)
        self.assertIsInstance(self.executor.safe_operations, list)
        self.assertIsInstance(self.executor.forbidden_operations, list)
        self.assertGreater(self.executor.max_operations, 0)
        self.assertGreater(self.executor.timeout_seconds, 0)
    
    def test_execute_operations_simple_filter(self):
        """Test executing simple filtering operations"""
        operations = [
            "df_filtered = df[df['category'] == 'Food']",
            "result = df_filtered"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertIn('result', result)
        self.assertIn('execution_history', result)
        
        # Check that filtering worked
        result_data = result['result']
        self.assertEqual(len(result_data), 2)  # Should have 2 Food transactions
    
    def test_execute_operations_groupby_aggregation(self):
        """Test executing groupby and aggregation operations"""
        operations = [
            "df_grouped = df.groupby('category')['amount'].sum()",
            "result = df_grouped"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        
        # Check result type
        self.assertEqual(result.get('result_type'), 'series')
        
        # Check that aggregation worked
        result_data = result['result']
        self.assertIn('Food', result_data)
        self.assertIn('Shopping', result_data)
        self.assertIn('Transportation', result_data)
        self.assertIn('Entertainment', result_data)
    
    def test_execute_operations_date_operations(self):
        """Test executing date-based operations"""
        operations = [
            "df['month'] = pd.to_datetime(df['date']).dt.month",
            "df_grouped = df.groupby('month')['amount'].sum()",
            "result = df_grouped.idxmax()"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertEqual(result.get('result_type'), 'single_value')
        
        # Check that result is a month number
        result_value = result['result']
        self.assertIsInstance(result_value, (int, np.integer))
        self.assertGreaterEqual(result_value, 1)
        self.assertLessEqual(result_value, 12)
    
    def test_execute_operations_complex_query(self):
        """Test executing complex query operations"""
        operations = [
            "df_filtered = df[df['category'] == 'Food']",
            "df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month",
            "df_grouped = df_filtered.groupby('month')['amount'].sum()",
            "result = df_grouped.idxmax()"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertEqual(result.get('result_type'), 'single_value')
        
        # Check that result is a month number
        result_value = result['result']
        self.assertIsInstance(result_value, (int, np.integer))
    
    def test_execute_operations_with_result_variable(self):
        """Test operations that use the 'result' variable"""
        operations = [
            "df_filtered = df[df['amount'] > 50]",
            "result = df_filtered['amount'].sum()"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertEqual(result.get('result_type'), 'scalar')
        
        # Check that result is a numeric value
        result_value = result['result']
        self.assertIsInstance(result_value, (int, float, np.number))
        self.assertGreater(result_value, 0)
    
    def test_validate_operations_too_many(self):
        """Test validation with too many operations"""
        operations = ["result = df.head()"] * 15  # More than max_operations
        
        with self.assertRaises(ValueError):
            self.executor.execute_operations(operations)
    
    def test_validate_operations_forbidden(self):
        """Test validation with forbidden operations"""
        operations = [
            "eval('dangerous_code')",
            "result = df.head()"
        ]
        
        with self.assertRaises(ValueError):
            self.executor.execute_operations(operations)
    
    def test_validate_operations_dangerous_patterns(self):
        """Test validation with dangerous patterns"""
        operations = [
            "df['__dangerous__'] = 'test'",
            "result = df.head()"
        ]
        
        with self.assertRaises(ValueError):
            self.executor.execute_operations(operations)
    
    def test_execute_operations_with_error(self):
        """Test executing operations that cause errors"""
        operations = [
            "df_filtered = df[df['nonexistent_column'] == 'value']",
            "result = df_filtered"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', False))
        self.assertIn('error', result)
    
    def test_format_results_single_value(self):
        """Test formatting single value results"""
        # Test with integer result
        operations = ["result = 42"]
        result = self.executor.execute_operations(operations)
        
        self.assertEqual(result.get('result_type'), 'scalar')
        self.assertEqual(result['result'], 42)
        
        # Test with float result
        operations = ["result = 42.5"]
        result = self.executor.execute_operations(operations)
        
        self.assertEqual(result.get('result_type'), 'scalar')
        self.assertEqual(result['result'], 42.5)
    
    def test_format_results_series(self):
        """Test formatting series results"""
        operations = [
            "df_grouped = df.groupby('category')['amount'].sum()",
            "result = df_grouped"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertEqual(result.get('result_type'), 'series')
        self.assertIsInstance(result['result'], dict)
        
        # Check that series data is properly formatted
        series_data = result['result']
        self.assertIn('Food', series_data)
        self.assertIn('Shopping', series_data)
    
    def test_format_results_dataframe(self):
        """Test formatting dataframe results"""
        operations = [
            "result = df.head(3)"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertEqual(result.get('result_type'), 'dataframe')
        self.assertIsInstance(result['result'], list)
        self.assertEqual(len(result['result']), 3)
    
    def test_get_execution_summary(self):
        """Test getting execution summary"""
        # Execute some operations first
        operations = [
            "df_filtered = df[df['category'] == 'Food']",
            "result = df_filtered"
        ]
        
        self.executor.execute_operations(operations)
        
        summary = self.executor.get_execution_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_operations', summary)
        self.assertIn('successful_operations', summary)
        self.assertIn('failed_operations', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('execution_history', summary)
        
        self.assertEqual(summary['total_operations'], 2)
        self.assertEqual(summary['successful_operations'], 2)
        self.assertEqual(summary['failed_operations'], 0)
        self.assertEqual(summary['success_rate'], 1.0)
    
    def test_reset_dataframe(self):
        """Test resetting dataframe to original state"""
        # Execute operations to modify current dataframe
        operations = ["df_filtered = df[df['category'] == 'Food']"]
        self.executor.execute_operations(operations)
        
        # Reset dataframe
        self.executor.reset_dataframe()
        
        # Check that current dataframe is back to original
        pd.testing.assert_frame_equal(self.executor.current_df, self.executor.original_df)
        self.assertEqual(len(self.executor.execution_history), 0)
    
    def test_get_current_dataframe(self):
        """Test getting current dataframe"""
        current_df = self.executor.get_current_dataframe()
        
        self.assertIsInstance(current_df, pd.DataFrame)
        pd.testing.assert_frame_equal(current_df, self.executor.current_df)
        
        # Check that it's a copy, not a reference
        self.assertIsNot(current_df, self.executor.current_df)


class TestSafePandasExecutor(unittest.TestCase):
    """Test the Safe Pandas Executor"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test dataframe
        self.test_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01']),
            'description': ['Amazon Purchase', 'Starbucks Coffee', 'Uber Ride'],
            'amount': [100.50, 5.25, 25.00],
            'category': ['Shopping', 'Food', 'Transportation'],
            'tab_name': ['Citi', 'Citi', 'Chase']
        })
        
        self.executor = SafePandasExecutor(self.test_data)
    
    def test_initialization(self):
        """Test safe executor initialization"""
        self.assertIsInstance(self.executor.max_dataframe_size, int)
        self.assertIsInstance(self.executor.max_memory_usage, int)
        self.assertGreater(self.executor.max_dataframe_size, 0)
        self.assertGreater(self.executor.max_memory_usage, 0)
    
    def test_execute_operations_with_size_monitoring(self):
        """Test operations with size monitoring"""
        operations = [
            "df_filtered = df[df['category'] == 'Food']",
            "result = df_filtered"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
    
    def test_validate_operations_with_warnings(self):
        """Test validation with warnings for many operations"""
        operations = [
            "df_filtered = df[df['category'] == 'Food']",
            "df_filtered2 = df_filtered[df_filtered['amount'] > 10]",
            "df_filtered3 = df_filtered2[df_filtered2['amount'] < 100]",
            "df_filtered4 = df_filtered3[df_filtered3['tab_name'] == 'Citi']",
            "df_filtered5 = df_filtered4[df_filtered4['date'] > '2024-01-01']",
            "result = df_filtered5"
        ]
        
        # Should not raise an exception but might log warnings
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))


class TestPandasExecutorIntegration(unittest.TestCase):
    """Integration tests for pandas executor"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create larger test dataset
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        self.test_data = pd.DataFrame({
            'date': np.random.choice(dates, 100),
            'description': np.random.choice(['Amazon', 'Starbucks', 'Uber', 'Netflix', 'Whole Foods'], 100),
            'amount': np.random.uniform(10, 500, 100),
            'category': np.random.choice(['Shopping', 'Food', 'Transportation', 'Entertainment'], 100),
            'tab_name': np.random.choice(['Citi', 'Chase', 'Cash'], 100)
        })
        
        self.executor = PandasExecutor(self.test_data)
    
    def test_complex_quarterly_analysis(self):
        """Test complex quarterly analysis operations"""
        operations = [
            "df_filtered = df[df['category'] == 'Shopping']",
            "df_filtered['quarter'] = pd.to_datetime(df_filtered['date']).dt.quarter",
            "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
            "result = df_grouped.idxmax()"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertEqual(result.get('result_type'), 'single_value')
        
        # Check that result is a quarter number
        result_value = result['result']
        self.assertIsInstance(result_value, (int, np.integer))
        self.assertGreaterEqual(result_value, 1)
        self.assertLessEqual(result_value, 4)
    
    def test_vendor_analysis(self):
        """Test vendor analysis operations"""
        operations = [
            "df_grouped = df.groupby('description')['amount'].sum()",
            "df_sorted = df_grouped.sort_values(ascending=False)",
            "result = df_sorted.head(3)"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertEqual(result.get('result_type'), 'series')
        
        # Check that result is a dictionary with top vendors
        result_data = result['result']
        self.assertIsInstance(result_data, dict)
        self.assertLessEqual(len(result_data), 3)
    
    def test_category_comparison(self):
        """Test category comparison operations"""
        operations = [
            "df_grouped = df.groupby('category')['amount'].agg(['sum', 'mean', 'count'])",
            "result = df_grouped"
        ]
        
        result = self.executor.execute_operations(operations)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertEqual(result.get('result_type'), 'dataframe')
        
        # Check that result contains multiple aggregations
        result_data = result['result']
        self.assertIsInstance(result_data, list)
        self.assertGreater(len(result_data), 0)


if __name__ == '__main__':
    unittest.main() 