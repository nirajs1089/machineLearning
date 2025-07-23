"""
Query Translator for converting natural language to pandas operations
"""

import json
import logging
from typing import Dict, List, Optional, Any
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import QUERY_TRANSLATION_CONFIG, ERROR_MESSAGES
from chatbot.llm_interface import LLMInterface


class QueryTranslator:
    """
    Translate natural language queries to pandas operations
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize the query translator
        
        Args:
            llm_interface: LLMInterface instance for query translation
        """
        self.llm_interface = llm_interface
        self.system_prompt = QUERY_TRANSLATION_CONFIG["system_prompt"]
        self.example_queries = QUERY_TRANSLATION_CONFIG["example_queries"]
    
    def translate_query(self, user_query: str) -> Dict[str, Any]:
        """
        Translate a natural language query to pandas operations
        
        Args:
            user_query: Natural language query string
            
        Returns:
            Dictionary with pandas operations and metadata
        """
        try:
            # Create the translation prompt
            prompt = self._create_translation_prompt(user_query)
            
            # Get translation from LLM
            response = self.llm_interface.generate_translation(prompt)
            
            # Parse the response
            translation_result = self._parse_translation_response(response)
            
            # Ensure the final operation assigns to 'result'
            translation_result['pandas_operations'] = self._ensure_result_assignment(
                translation_result.get('pandas_operations', [])
            )
            
            # Validate the translation
            self._validate_translation(translation_result)
            
            logger.info(f"Successfully translated query: {user_query}")
            return translation_result
            
        except Exception as e:
            logger.error(f"Error translating query '{user_query}': {e}")
            return {
                "error": str(e),
                "pandas_operations": [],
                "explanation": "Unable to translate query"
            }
    
    def _create_translation_prompt(self, user_query: str) -> str:
        """
        Create a prompt for query translation
        
        Args:
            user_query: Natural language query
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system prompt
        prompt_parts.append(self.system_prompt)
        
        # Add example queries
        prompt_parts.append("\nExample translations:")
        for query_type, example in self.example_queries.items():
            prompt_parts.append(f"\nQuery: {example['query']}")
            prompt_parts.append("Operations:")
            for operation in example['operations']:
                prompt_parts.append(f"  {operation}")
        
        # Add the user query
        prompt_parts.append(f"\n\nNow translate this query:\nQuery: {user_query}")
        prompt_parts.append("\nReturn only a JSON response in this format:")
        prompt_parts.append("""
{
  "pandas_operations": [
    "operation1",
    "operation2",
    "operation3"
  ],
  "explanation": "Brief explanation of what the operations do"
}
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_translation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured format
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed translation result
        """
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Ensure required fields
            if 'pandas_operations' not in parsed:
                parsed['pandas_operations'] = []
            
            if 'explanation' not in parsed:
                parsed['explanation'] = "No explanation provided"
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Error parsing translation response: {e}")
            raise
    
    def _validate_translation(self, translation_result: Dict[str, Any]) -> None:
        """
        Validate the translation result
        
        Args:
            translation_result: Translation result to validate
            
        Raises:
            ValueError: If validation fails
        """
        if 'pandas_operations' not in translation_result:
            raise ValueError("Missing pandas_operations in translation result")
        
        if not isinstance(translation_result['pandas_operations'], list):
            raise ValueError("pandas_operations must be a list")
        
        if len(translation_result['pandas_operations']) == 0:
            raise ValueError("No pandas operations provided")
        
        # Check for forbidden operations
        forbidden_ops = ['eval', 'exec', 'system', 'subprocess', 'os.', 'import']
        operations_text = ' '.join(translation_result['pandas_operations']).lower()
        
        for forbidden_op in forbidden_ops:
            if forbidden_op in operations_text:
                raise ValueError(f"Forbidden operation detected: {forbidden_op}")
    
    def get_supported_operations(self) -> List[str]:
        """
        Get list of supported pandas operations
        
        Returns:
            List of supported operations
        """
        return [
            "DataFrame filtering (df[condition])",
            "GroupBy operations (df.groupby())",
            "Aggregation functions (.sum(), .mean(), .count(), .max(), .min())",
            "Sorting (.sort_values())",
            "Date operations (pd.to_datetime(), .dt.month, .dt.quarter, .dt.year)",
            "Column operations (.rename(), .reset_index())",
            "Selection operations (.head(), .tail())"
        ]
    
    def get_example_queries(self) -> Dict[str, str]:
        """
        Get example queries for reference
        
        Returns:
            Dictionary of example queries
        """
        return {
            query_type: example["query"] 
            for query_type, example in self.example_queries.items()
        }

    def _ensure_result_assignment(self, operations: List[str]) -> List[str]:
        """
        Ensure the final operation assigns to the 'result' variable.
        If not, append an assignment to 'result'.
        """
        if not operations:
            return operations
        last_op = operations[-1].strip()
        # If the last operation already assigns to 'result', do nothing
        if last_op.startswith('result ='):
            return operations
        # If the last operation is a variable assignment, assign it to result
        if '=' in last_op:
            var = last_op.split('=')[0].strip()
            # If it's not 'result', add 'result = var' as the last step
            if var != 'result':
                return operations + [f"result = {var}"]
        # If the last operation is an expression, assign it to result
        return operations[:-1] + [f"result = {last_op}"]


class MockQueryTranslator:
    """
    Mock query translator for testing purposes
    """
    
    def __init__(self):
        """Initialize mock translator"""
        self.mock_translations = {
            "which quarter was the most expensive in the travel category?": {
                "pandas_operations": [
                    "df_filtered = df[df['category'] == 'travel']",
                    "df_filtered['quarter'] = pd.to_datetime(df_filtered['date']).dt.quarter",
                    "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
                    "result = df_grouped.idxmax()"
                ],
                "explanation": "Filter by travel category, group by quarter, sum amounts, find quarter with max sum"
            },
            "which is the most expensive month in the travel category?": {
                "pandas_operations": [
                    "df_filtered = df[df['category'] == 'travel']",
                    "df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month",
                    "df_grouped = df_filtered.groupby('month')['amount'].sum()",
                    "result = df_grouped.idxmax()"
                ],
                "explanation": "Filter by travel category, group by month, sum amounts, find month with max sum"
            },
            "which vendor did i spend the most money with?": {
                "pandas_operations": [
                    "df_grouped = df.groupby('description')['amount'].sum()",
                    "result = df_grouped.idxmax()"
                ],
                "explanation": "Group by description, sum amounts, find description with max sum"
            }
        }
    
    def translate_query(self, user_query: str) -> Dict[str, Any]:
        """
        Mock translation of natural language query
        
        Args:
            user_query: Natural language query
            
        Returns:
            Mock translation result
        """
        # Try to find exact match
        if user_query.lower() in self.mock_translations:
            return self.mock_translations[user_query.lower()]
        
        # Try to find partial match
        for key, value in self.mock_translations.items():
            if any(word in user_query.lower() for word in key.split()):
                return value
        
        # Return default translation
        return {
            "pandas_operations": [
                "result = df.head()"
            ],
            "explanation": "Default operation - show first few rows"
        }
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations"""
        return ["Mock operations for testing"]
    
    def get_example_queries(self) -> Dict[str, str]:
        """Get example queries"""
        return {
            "quarterly_analysis": "which quarter was the most expensive in the travel category?",
            "monthly_analysis": "which is the most expensive month in the travel category?",
            "vendor_analysis": "which vendor did i spend the most money with?"
        }


logger = logging.getLogger(__name__) 