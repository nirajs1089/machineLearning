"""
Test file for LLM Interface
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.llm_interface import LLMInterface, MockLLMInterface


class TestMockLLMInterface(unittest.TestCase):
    """Test the mock LLM interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.llm = MockLLMInterface()
    
    def test_initialization(self):
        """Test mock LLM initialization"""
        self.assertTrue(self.llm.available)
        self.assertIsInstance(self.llm.mock_translations, dict)
        self.assertGreater(len(self.llm.mock_translations), 0)
    
    def test_generate_translation_exact_match(self):
        """Test translation generation with exact match"""
        prompt = "which quarter was the most expensive in the travel category?"
        
        response = self.llm.generate_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('pandas_operations', response)
        self.assertIn('travel', response)
        self.assertIn('quarter', response)
        self.assertIn('groupby', response)
    
    def test_generate_translation_partial_match(self):
        """Test translation generation with partial match"""
        prompt = "what's the most expensive quarter for travel?"
        
        response = self.llm.generate_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('pandas_operations', response)
        
        # Should find a match based on keywords
        if 'travel' in prompt.lower() and 'quarter' in prompt.lower():
            self.assertIn('travel', response)
            self.assertIn('quarter', response)
    
    def test_generate_translation_no_match(self):
        """Test translation generation with no match"""
        prompt = "what is the weather like today?"
        
        response = self.llm.generate_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('pandas_operations', response)
        self.assertIn('df.head()', response)  # Default operation
    
    def test_generate_response(self):
        """Test response generation"""
        query = "which quarter was the most expensive in the travel category?"
        formatted_result = {
            "response": "The most expensive quarter for travel was Q2 with $1,500",
            "result_type": "quarterly_series",
            "success": True
        }
        
        response = self.llm.generate_response(query, formatted_result)
        
        self.assertIsInstance(response, str)
        self.assertIn("most expensive quarter", response)
    
    def test_generate_response_with_error(self):
        """Test response generation with error result"""
        query = "test query"
        formatted_result = {
            "response": "Error occurred",
            "success": False
        }
        
        response = self.llm.generate_response(query, formatted_result)
        
        self.assertIsInstance(response, str)
        self.assertIn("Error occurred", response)
    
    def test_generate_follow_up_questions(self):
        """Test follow-up question generation"""
        query = "which quarter was the most expensive in the travel category?"
        formatted_result = {
            "response": "The most expensive quarter for travel was Q2",
            "result_type": "quarterly_series",
            "success": True
        }
        
        follow_ups = self.llm.generate_follow_up_questions(query, formatted_result)
        
        self.assertIsInstance(follow_ups, list)
        self.assertGreater(len(follow_ups), 0)
        self.assertLessEqual(len(follow_ups), 3)
        
        for question in follow_ups:
            self.assertIsInstance(question, str)
            self.assertIn('?', question)
    
    def test_test_connection(self):
        """Test connection test"""
        result = self.llm.test_connection()
        self.assertTrue(result)


class TestLLMInterface(unittest.TestCase):
    """Test the real LLM interface with mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_api_key = "test_api_key"
        self.mock_model = "gpt-4"
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_initialization_success(self, mock_openai):
        """Test successful initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        self.assertEqual(llm.api_key, self.mock_api_key)
        self.assertEqual(llm.model, self.mock_model)
        self.assertTrue(llm.available)
        self.assertIsNotNone(llm.client)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_initialization_failure(self, mock_openai):
        """Test initialization failure"""
        mock_openai.side_effect = Exception("OpenAI import failed")
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        self.assertFalse(llm.available)
        self.assertIsNone(llm.client)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_generate_translation_success(self, mock_openai):
        """Test successful translation generation"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']",
    "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
    "result = df_grouped.idxmax()"
  ],
  "explanation": "Filter by travel category, group by quarter, sum amounts"
}'''
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        prompt = "which quarter was the most expensive in the travel category?"
        response = llm.generate_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('pandas_operations', response)
        self.assertIn('travel', response)
        
        # Verify the API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], self.mock_model)
        self.assertEqual(call_args[1]['temperature'], llm.temperature)
        self.assertEqual(call_args[1]['max_tokens'], llm.max_tokens)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_generate_translation_api_error(self, mock_openai):
        """Test translation generation with API error"""
        # Mock the OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        prompt = "test query"
        response = llm.generate_translation(prompt)
        
        # Should return fallback translation
        self.assertIsInstance(response, str)
        self.assertIn('pandas_operations', response)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_generate_response_success(self, mock_openai):
        """Test successful response generation"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The most expensive quarter for travel was Q2 with $1,500."
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        query = "which quarter was the most expensive in the travel category?"
        formatted_result = {
            "response": "Q2",
            "result_type": "quarterly_series",
            "success": True
        }
        
        response = llm.generate_response(query, formatted_result)
        
        self.assertIsInstance(response, str)
        self.assertIn("most expensive quarter", response)
        
        # Verify the API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], self.mock_model)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_generate_response_with_conversation_history(self, mock_openai):
        """Test response generation with conversation history"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Based on our previous conversation, Q2 was the most expensive."
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        query = "which quarter was the most expensive in the travel category?"
        formatted_result = {
            "response": "Q2",
            "result_type": "quarterly_series",
            "success": True
        }
        conversation_history = [
            {"role": "user", "content": "What are my spending patterns?"},
            {"role": "assistant", "content": "Your spending varies by quarter."}
        ]
        
        response = llm.generate_response(query, formatted_result, conversation_history)
        
        self.assertIsInstance(response, str)
        
        # Verify the API was called with conversation history
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertGreater(len(messages), 2)  # Should include system, user, and history
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_generate_follow_up_questions_success(self, mock_openai):
        """Test successful follow-up question generation"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
1. What was the least expensive quarter?
2. How does this compare to last year?
3. What caused the high spending in Q2?
"""
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        query = "which quarter was the most expensive in the travel category?"
        formatted_result = {
            "response": "Q2 was the most expensive quarter",
            "result_type": "quarterly_series",
            "success": True
        }
        
        follow_ups = llm.generate_follow_up_questions(query, formatted_result)
        
        self.assertIsInstance(follow_ups, list)
        self.assertGreater(len(follow_ups), 0)
        self.assertLessEqual(len(follow_ups), 3)
        
        for question in follow_ups:
            self.assertIsInstance(question, str)
            self.assertIn('?', question)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_test_connection_success(self, mock_openai):
        """Test successful connection test"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        result = llm.test_connection()
        self.assertTrue(result)
    
    @patch('chatbot.llm_interface.OpenAI')
    def test_test_connection_failure(self, mock_openai):
        """Test connection test failure"""
        # Mock the OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        mock_openai.return_value = mock_client
        
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        
        result = llm.test_connection()
        self.assertFalse(result)
    
    def test_fallback_translation_travel_quarter(self):
        """Test fallback translation for travel quarter query"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        prompt = "which quarter was the most expensive in the travel category?"
        response = llm._generate_fallback_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('travel', response)
        self.assertIn('quarter', response)
        self.assertIn('groupby', response)
    
    def test_fallback_translation_travel_month(self):
        """Test fallback translation for travel month query"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        prompt = "which is the most expensive month in the travel category?"
        response = llm._generate_fallback_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('travel', response)
        self.assertIn('month', response)
        self.assertIn('groupby', response)
    
    def test_fallback_translation_generic(self):
        """Test fallback translation for generic query"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        prompt = "what is the weather like?"
        response = llm._generate_fallback_translation(prompt)
        
        self.assertIsInstance(response, str)
        self.assertIn('df.head()', response)
    
    def test_fallback_response_success(self):
        """Test fallback response for successful result"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        query = "which quarter was the most expensive?"
        formatted_result = {
            "response": "Q2 was the most expensive quarter",
            "result_type": "quarterly_series",
            "success": True
        }
        
        response = llm._generate_fallback_response(query, formatted_result)
        
        self.assertIsInstance(response, str)
        self.assertIn("Q2 was the most expensive quarter", response)
        self.assertIn("spending pattern by quarter", response)
    
    def test_fallback_response_error(self):
        """Test fallback response for error result"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        query = "test query"
        formatted_result = {
            "response": "Error occurred",
            "success": False
        }
        
        response = llm._generate_fallback_response(query, formatted_result)
        
        self.assertIsInstance(response, str)
        self.assertIn("I'm sorry", response)
        self.assertIn("try again", response)
    
    def test_fallback_follow_ups_quarterly(self):
        """Test fallback follow-up questions for quarterly analysis"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        query = "which quarter was the most expensive?"
        formatted_result = {
            "response": "Q2 was the most expensive",
            "result_type": "quarterly_series",
            "success": True
        }
        
        follow_ups = llm._generate_fallback_follow_ups(query, formatted_result)
        
        self.assertIsInstance(follow_ups, list)
        self.assertEqual(len(follow_ups), 3)
        self.assertIn("least expensive quarter", follow_ups[0])
        self.assertIn("compare to last year", follow_ups[1])
    
    def test_fallback_follow_ups_vendor(self):
        """Test fallback follow-up questions for vendor analysis"""
        llm = LLMInterface(api_key=self.mock_api_key, model=self.mock_model)
        llm.available = False  # Force fallback
        
        query = "which vendor did I spend the most with?"
        formatted_result = {
            "response": "Amazon was your top vendor",
            "result_type": "vendor",
            "success": True
        }
        
        follow_ups = llm._generate_fallback_follow_ups(query, formatted_result)
        
        self.assertIsInstance(follow_ups, list)
        self.assertEqual(len(follow_ups), 3)
        self.assertIn("categories", follow_ups[0])
        self.assertIn("vendor", follow_ups[1])


class TestLLMInterfaceIntegration(unittest.TestCase):
    """Integration tests for LLM interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.llm = MockLLMInterface()  # Use mock for integration tests
    
    def test_complete_translation_workflow(self):
        """Test complete translation workflow"""
        query = "which quarter was the most expensive in the travel category?"
        
        # Step 1: Generate translation
        translation = self.llm.generate_translation(query)
        
        self.assertIsInstance(translation, str)
        self.assertIn('pandas_operations', translation)
        
        # Step 2: Generate response
        formatted_result = {
            "response": "Q2 was the most expensive quarter",
            "result_type": "quarterly_series",
            "success": True
        }
        
        response = self.llm.generate_response(query, formatted_result)
        
        self.assertIsInstance(response, str)
        self.assertIn("most expensive quarter", response)
        
        # Step 3: Generate follow-up questions
        follow_ups = self.llm.generate_follow_up_questions(query, formatted_result)
        
        self.assertIsInstance(follow_ups, list)
        self.assertGreater(len(follow_ups), 0)
    
    def test_error_handling_workflow(self):
        """Test error handling workflow"""
        query = "invalid query"
        formatted_result = {
            "response": "Error occurred",
            "success": False
        }
        
        # Should handle errors gracefully
        response = self.llm.generate_response(query, formatted_result)
        self.assertIsInstance(response, str)
        
        follow_ups = self.llm.generate_follow_up_questions(query, formatted_result)
        self.assertIsInstance(follow_ups, list)


if __name__ == '__main__':
    unittest.main() 