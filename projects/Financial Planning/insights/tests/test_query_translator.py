"""
Test file for Query Translator
"""

import unittest
import json
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.query_translator import QueryTranslator, MockQueryTranslator
from chatbot.llm_interface import MockLLMInterface


class TestMockQueryTranslator(unittest.TestCase):
    """Test the mock query translator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.translator = MockQueryTranslator()
    
    def test_translate_query_exact_match(self):
        """Test translation with exact query match"""
        query = "which quarter was the most expensive in the travel category?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        self.assertIn('explanation', result)
        
        operations = result['pandas_operations']
        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
        
        # Check that operations contain expected patterns
        operations_text = ' '.join(operations).lower()
        self.assertIn('travel', operations_text)
        self.assertIn('quarter', operations_text)
        self.assertIn('groupby', operations_text)
    
    def test_translate_query_partial_match(self):
        """Test translation with partial query match"""
        query = "what's the most expensive quarter for travel?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        self.assertIn('explanation', result)
        
        # Should find a match based on keywords
        operations = result['pandas_operations']
        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
    
    def test_translate_query_no_match(self):
        """Test translation with no matching query"""
        query = "what is the weather like today?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        self.assertIn('explanation', result)
        
        # Should return default operation
        operations = result['pandas_operations']
        self.assertIsInstance(operations, list)
        self.assertIn('df.head()', operations[0])
    
    def test_get_supported_operations(self):
        """Test getting supported operations"""
        operations = self.translator.get_supported_operations()
        
        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
        self.assertIn("Mock operations for testing", operations)
    
    def test_get_example_queries(self):
        """Test getting example queries"""
        examples = self.translator.get_example_queries()
        
        self.assertIsInstance(examples, dict)
        self.assertGreater(len(examples), 0)
        
        expected_keys = ['quarterly_analysis', 'monthly_analysis', 'vendor_analysis']
        for key in expected_keys:
            self.assertIn(key, examples)
            self.assertIsInstance(examples[key], str)


class TestQueryTranslator(unittest.TestCase):
    """Test the real query translator with mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = MockLLMInterface()
        self.translator = QueryTranslator(self.mock_llm)
    
    def test_initialization(self):
        """Test translator initialization"""
        self.assertIsNotNone(self.translator.llm_interface)
        self.assertIsNotNone(self.translator.system_prompt)
        self.assertIsNotNone(self.translator.example_queries)
    
    def test_translate_query_success(self):
        """Test successful query translation"""
        query = "which quarter was the most expensive in the travel category?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        self.assertIn('explanation', result)
        self.assertNotIn('error', result)
        
        operations = result['pandas_operations']
        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
    
    def test_translate_query_with_error(self):
        """Test query translation with LLM error"""
        # Create a mock LLM that raises an exception
        mock_llm = Mock()
        mock_llm.generate_translation.side_effect = Exception("LLM Error")
        
        translator = QueryTranslator(mock_llm)
        query = "test query"
        
        result = translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('pandas_operations', result)
        self.assertEqual(result['pandas_operations'], [])
    
    def test_create_translation_prompt(self):
        """Test creating translation prompt"""
        query = "which quarter was the most expensive in the travel category?"
        
        prompt = self.translator._create_translation_prompt(query)
        
        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("pandas_operations", prompt)
        self.assertIn("Available columns", prompt)
        self.assertIn("Example translations", prompt)
    
    def test_parse_translation_response_valid_json(self):
        """Test parsing valid JSON response"""
        valid_response = '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']",
    "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
    "result = df_grouped.idxmax()"
  ],
  "explanation": "Filter by travel category, group by quarter, sum amounts"
}'''
        
        result = self.translator._parse_translation_response(valid_response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        self.assertIn('explanation', result)
        
        operations = result['pandas_operations']
        self.assertIsInstance(operations, list)
        self.assertEqual(len(operations), 3)
    
    def test_parse_translation_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        invalid_response = "This is not valid JSON"
        
        with self.assertRaises(ValueError):
            self.translator._parse_translation_response(invalid_response)
    
    def test_parse_translation_response_missing_fields(self):
        """Test parsing response with missing fields"""
        response_with_missing_fields = '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']"
  ]
}'''
        
        result = self.translator._parse_translation_response(response_with_missing_fields)
        
        self.assertIn('pandas_operations', result)
        self.assertIn('explanation', result)
        self.assertEqual(result['explanation'], "No explanation provided")
    
    def test_validate_translation_valid(self):
        """Test validation of valid translation"""
        valid_translation = {
            'pandas_operations': [
                'df_filtered = df[df["category"] == "travel"]',
                'result = df_filtered.head()'
            ],
            'explanation': 'Test explanation'
        }
        
        # Should not raise an exception
        self.translator._validate_translation(valid_translation)
    
    def test_validate_translation_missing_operations(self):
        """Test validation with missing operations"""
        invalid_translation = {
            'explanation': 'Test explanation'
        }
        
        with self.assertRaises(ValueError):
            self.translator._validate_translation(invalid_translation)
    
    def test_validate_translation_empty_operations(self):
        """Test validation with empty operations"""
        invalid_translation = {
            'pandas_operations': [],
            'explanation': 'Test explanation'
        }
        
        with self.assertRaises(ValueError):
            self.translator._validate_translation(invalid_translation)
    
    def test_validate_translation_forbidden_operations(self):
        """Test validation with forbidden operations"""
        invalid_translation = {
            'pandas_operations': [
                'eval("dangerous_code")',
                'df_filtered = df[df["category"] == "travel"]'
            ],
            'explanation': 'Test explanation'
        }
        
        with self.assertRaises(ValueError):
            self.translator._validate_translation(invalid_translation)
    
    def test_get_supported_operations(self):
        """Test getting supported operations"""
        operations = self.translator.get_supported_operations()
        
        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
        
        # Check for expected operation types
        operations_text = ' '.join(operations).lower()
        self.assertIn('filtering', operations_text)
        self.assertIn('groupby', operations_text)
        self.assertIn('aggregation', operations_text)
        self.assertIn('sorting', operations_text)
    
    def test_get_example_queries(self):
        """Test getting example queries"""
        examples = self.translator.get_example_queries()
        
        self.assertIsInstance(examples, dict)
        self.assertGreater(len(examples), 0)
        
        # Check for expected example types
        expected_types = ['quarterly_analysis', 'monthly_analysis', 'vendor_analysis']
        for query_type in expected_types:
            self.assertIn(query_type, examples)
            self.assertIsInstance(examples[query_type], str)
            self.assertIn('?', examples[query_type])


class TestQueryTranslationIntegration(unittest.TestCase):
    """Integration tests for query translation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = MockLLMInterface()
        self.translator = QueryTranslator(self.mock_llm)
    
    def test_travel_quarterly_analysis(self):
        """Test translation for travel quarterly analysis"""
        query = "which quarter was the most expensive in the travel category?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        
        operations = result['pandas_operations']
        operations_text = ' '.join(operations).lower()
        
        # Check for expected operations
        self.assertIn('travel', operations_text)
        self.assertIn('quarter', operations_text)
        self.assertIn('groupby', operations_text)
        self.assertIn('sum', operations_text)
        self.assertIn('idxmax', operations_text)
    
    def test_travel_monthly_analysis(self):
        """Test translation for travel monthly analysis"""
        query = "which is the most expensive month in the travel category?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        
        operations = result['pandas_operations']
        operations_text = ' '.join(operations).lower()
        
        # Check for expected operations
        self.assertIn('travel', operations_text)
        self.assertIn('month', operations_text)
        self.assertIn('groupby', operations_text)
        self.assertIn('sum', operations_text)
        self.assertIn('idxmax', operations_text)
    
    def test_vendor_analysis(self):
        """Test translation for vendor analysis"""
        query = "which vendor did I spend the most money with?"
        
        result = self.translator.translate_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pandas_operations', result)
        
        operations = result['pandas_operations']
        operations_text = ' '.join(operations).lower()
        
        # Check for expected operations
        self.assertIn('description', operations_text)
        self.assertIn('groupby', operations_text)
        self.assertIn('sum', operations_text)
        self.assertIn('idxmax', operations_text)


if __name__ == '__main__':
    unittest.main() 