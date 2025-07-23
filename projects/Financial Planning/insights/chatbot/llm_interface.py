"""
LLM Interface for query translation and response generation
"""

import logging
from typing import Dict, List, Optional, Any
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OPENAI_CONFIG, CHATBOT_CONFIG, ERROR_MESSAGES

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Interface for communicating with Large Language Models
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM interface
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.api_key = api_key or OPENAI_CONFIG["api_key"]
        self.model = model or OPENAI_CONFIG["model"]
        self.temperature = OPENAI_CONFIG["temperature"]
        self.max_tokens = OPENAI_CONFIG["max_tokens"]
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.available = True
            logger.info("LLM interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM interface: {e}")
            self.available = False
    
    def generate_translation(self, prompt: str) -> str:
        """
        Generate pandas operations translation from natural language
        
        Args:
            prompt: Translation prompt
            
        Returns:
            Translation response from LLM
        """
        if not self.available:
            return self._generate_fallback_translation(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query translation expert. Convert natural language queries about financial transactions into pandas operations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating translation: {e}")
            return self._generate_fallback_translation(prompt)
    
    def generate_response(self, 
                         query: str, 
                         formatted_result: Dict[str, Any],
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a natural language response based on formatted results
        
        Args:
            query: Original user query
            formatted_result: Formatted execution results
            conversation_history: Previous conversation context
            
        Returns:
            Natural language response
        """
        if not self.available:
            return self._generate_fallback_response(query, formatted_result)
        
        try:
            # Create the prompt
            prompt = self._create_response_prompt(query, formatted_result, conversation_history)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CHATBOT_CONFIG["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(query, formatted_result)
    
    def generate_follow_up_questions(self, 
                                   query: str, 
                                   formatted_result: Dict[str, Any]) -> List[str]:
        """
        Generate follow-up questions based on the result
        
        Args:
            query: Original user query
            formatted_result: Formatted execution results
            
        Returns:
            List of suggested follow-up questions
        """
        if not self.available:
            return self._generate_fallback_follow_ups(query, formatted_result)
        
        try:
            prompt = f"""
Based on this analysis:
Query: {query}
Result: {formatted_result.get('response', '')}

Generate 3 relevant follow-up questions that would provide additional insights. 
Make them specific and actionable. Format as a simple list.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse the response into a list
            content = response.choices[0].message.content.strip()
            questions = [q.strip() for q in content.split('\n') if q.strip() and '?' in q]
            
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return self._generate_fallback_follow_ups(query, formatted_result)
    
    def _create_response_prompt(self, 
                               query: str, 
                               formatted_result: Dict[str, Any],
                               conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Create a prompt for response generation
        
        Args:
            query: Original user query
            formatted_result: Formatted execution results
            conversation_history: Previous conversation context
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add context about the user's question
        prompt_parts.append(f"User Question: {query}")
        
        # Add analysis results
        if formatted_result.get("success", False):
            prompt_parts.append("Analysis Results:")
            prompt_parts.append(f"Result Type: {formatted_result.get('result_type', 'unknown')}")
            
            # Add specific details based on result type
            result_type = formatted_result.get("result_type", "")
            if result_type == "monthly_series":
                prompt_parts.append(f"Top Month: {formatted_result.get('top_month', '')}")
                prompt_parts.append(f"Amount: ${formatted_result.get('top_amount', 0):,.2f}")
            elif result_type == "quarterly_series":
                prompt_parts.append(f"Top Quarter: {formatted_result.get('top_quarter', '')}")
                prompt_parts.append(f"Amount: ${formatted_result.get('top_amount', 0):,.2f}")
            elif result_type == "vendor":
                prompt_parts.append(f"Top Vendor: {formatted_result.get('string_value', '')}")
            elif result_type == "category":
                prompt_parts.append(f"Top Category: {formatted_result.get('string_value', '')}")
            elif result_type == "dataframe":
                # For DataFrame results, include a structured summary
                text_list = formatted_result.get("text_list")
                if text_list:
                    prompt_parts.append("Top Results:")
                    for line in text_list:
                        prompt_parts.append(line)
                else:
                    # Fallback: show first 3 rows as JSON
                    table = formatted_result.get("table", [])
                    for i, row in enumerate(table[:3]):
                        prompt_parts.append(f"Row {i+1}: {row}")
            elif result_type == "series" or result_type == "generic_series":
                # For series/groupby, include all key-value pairs
                all_data = formatted_result.get("all_data", {})
                if all_data:
                    prompt_parts.append("All Results:")
                    for k, v in all_data.items():
                        prompt_parts.append(f"{k}: {v}")
            elif result_type == "dataframe_with_amount":
                amount = formatted_result.get("amount")
                prompt_parts.append(f"Amount: ${amount:,.2f}")
            elif result_type == "single_value" or result_type == "scalar" or result_type == "amount":
                prompt_parts.append(f"Value: {formatted_result.get('numeric_value', formatted_result.get('value', ''))}")
            else:
                # Fallback: show the response string
                prompt_parts.append(f"Response: {formatted_result.get('response', '')}")
        else:
            prompt_parts.append("Analysis failed or no data found.")
        
        # Add conversation history for context
        if conversation_history:
            prompt_parts.append("Previous Conversation Context:")
            for msg in conversation_history[-3:]:  # Last 3 messages
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
        
        # Add instructions
        prompt_parts.append("""
Please provide a clear, helpful response that:
1. Directly answers the user's question
2. Includes specific numbers and percentages when available
3. Provides actionable insights
4. Uses a friendly, conversational tone
5. Suggests follow-up questions if relevant
""")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_fallback_translation(self, prompt: str) -> str:
        """
        Generate fallback translation when LLM is not available
        
        Args:
            prompt: Translation prompt
            
        Returns:
            Fallback translation response
        """
        # Simple fallback based on common patterns
        if "travel" in prompt.lower() and "quarter" in prompt.lower():
            return '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']",
    "df_filtered['quarter'] = pd.to_datetime(df_filtered['date']).dt.quarter",
    "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
    "result = df_grouped.idxmax()"
  ],
  "explanation": "Filter by travel category, group by quarter, sum amounts, find quarter with max sum"
}'''
        elif "travel" in prompt.lower() and "month" in prompt.lower():
            return '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']",
    "df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month",
    "df_grouped = df_filtered.groupby('month')['amount'].sum()",
    "result = df_grouped.idxmax()"
  ],
  "explanation": "Filter by travel category, group by month, sum amounts, find month with max sum"
}'''
        else:
            return '''{
  "pandas_operations": [
    "result = df.head()"
  ],
  "explanation": "Default operation - show first few rows"
}'''
    
    def _generate_fallback_response(self, query: str, formatted_result: Dict[str, Any]) -> str:
        """
        Generate fallback response when LLM is not available
        
        Args:
            query: Original user query
            formatted_result: Formatted execution results
            
        Returns:
            Fallback response
        """
        if not formatted_result.get("success", False):
            return "I'm sorry, I encountered an error while processing your request. Please try again."
        
        response = formatted_result.get("response", "Analysis completed.")
        
        # Add some context based on result type
        result_type = formatted_result.get("result_type", "")
        if result_type == "monthly_series":
            response += " This shows your spending pattern by month."
        elif result_type == "quarterly_series":
            response += " This shows your spending pattern by quarter."
        elif result_type == "vendor":
            response += " This identifies your top spending vendor."
        
        return response
    
    def _generate_fallback_follow_ups(self, query: str, formatted_result: Dict[str, Any]) -> List[str]:
        """
        Generate fallback follow-up questions
        
        Args:
            query: Original user query
            formatted_result: Formatted execution results
            
        Returns:
            List of fallback follow-up questions
        """
        result_type = formatted_result.get("result_type", "")
        
        if result_type == "monthly_series":
            return [
                "What was the least expensive month?",
                "How does this compare to last year?",
                "What caused the high spending in that month?"
            ]
        elif result_type == "quarterly_series":
            return [
                "What was the least expensive quarter?",
                "How does this compare to last year?",
                "What caused the high spending in that quarter?"
            ]
        elif result_type == "vendor":
            return [
                "What categories do I spend the most on?",
                "How much did I spend with that vendor?",
                "What other vendors do I spend a lot with?"
            ]
        else:
            return [
                "Can you show me the breakdown by month?",
                "What are my top spending categories?",
                "How does this compare to previous periods?"
            ]
    
    def test_connection(self) -> bool:
        """
        Test the connection to the LLM service
        
        Returns:
            True if connection is successful, False otherwise
        """
        if not self.available:
            return False
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False


class MockLLMInterface:
    """
    Mock LLM interface for testing purposes
    """
    
    def __init__(self):
        """Initialize mock interface"""
        self.available = True
        self.mock_translations = {
            "which quarter was the most expensive in the travel category?": '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']",
    "df_filtered['quarter'] = pd.to_datetime(df_filtered['date']).dt.quarter",
    "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
    "result = df_grouped.idxmax()"
  ],
  "explanation": "Filter by travel category, group by quarter, sum amounts, find quarter with max sum"
}''',
            "which is the most expensive month in the travel category?": '''{
  "pandas_operations": [
    "df_filtered = df[df['category'] == 'travel']",
    "df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month",
    "df_grouped = df_filtered.groupby('month')['amount'].sum()",
    "result = df_grouped.idxmax()"
  ],
  "explanation": "Filter by travel category, group by month, sum amounts, find month with max sum"
}'''
        }
    
    def generate_translation(self, prompt: str) -> str:
        """Mock translation generation"""
        # Try to find a matching query in the prompt
        for query, translation in self.mock_translations.items():
            if query in prompt.lower():
                return translation
        
        # Return default translation
        return '''{
  "pandas_operations": [
    "result = df.head()"
  ],
  "explanation": "Default operation - show first few rows"
}'''
    
    def generate_response(self, query: str, formatted_result: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Mock response generation"""
        return formatted_result.get("response", "Mock response generated.")
    
    def generate_follow_up_questions(self, query: str, formatted_result: Dict[str, Any]) -> List[str]:
        """Mock follow-up questions"""
        return [
            "What was the least expensive period?",
            "How does this compare to last year?",
            "What caused the high spending?"
        ]
    
    def test_connection(self) -> bool:
        """Mock connection test"""
        return True 