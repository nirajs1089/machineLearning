"""
Result Formatter for converting pandas execution results to user-friendly responses
"""

import logging
from typing import Dict, List, Optional, Any
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ERROR_MESSAGES

logger = logging.getLogger(__name__)


class ResultFormatter:
    """
    Format pandas execution results into user-friendly responses
    """
    
    def __init__(self):
        """Initialize the result formatter"""
        self.month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        
        self.quarter_names = {
            1: "Q1 (January-March)",
            2: "Q2 (April-June)", 
            3: "Q3 (July-September)",
            4: "Q4 (October-December)"
        }
    
    def format_execution_result(self, 
                               execution_result: Dict[str, Any], 
                               original_query: str,
                               translation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format execution result into user-friendly response
        
        Args:
            execution_result: Result from pandas execution
            original_query: Original user query
            translation_result: Translation result from LLM
            
        Returns:
            Formatted response dictionary
        """
        try:
            if not execution_result.get("success", False):
                return self._format_error_response(execution_result, original_query)
            
            result_type = execution_result.get("result_type", "unknown")
            result_value = execution_result.get("result")
            
            # Format based on result type
            if result_type == "single_value":
                formatted_response = self._format_single_value_result(result_value, original_query)
            elif result_type == "series":
                formatted_response = self._format_series_result(result_value, original_query)
            elif result_type == "dataframe":
                formatted_response = self._format_dataframe_result(result_value, original_query)
            elif result_type == "scalar":
                formatted_response = self._format_scalar_result(result_value, original_query)
            else:
                formatted_response = self._format_unknown_result(result_value, original_query)
            
            # Add metadata
            formatted_response.update({
                "original_query": original_query,
                "translation_explanation": translation_result.get("explanation", ""),
                "execution_summary": execution_result.get("execution_history", []),
                "success": True
            })
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
            return self._format_error_response({"error": str(e)}, original_query)
    
    def _format_single_value_result(self, result_value: Any, query: str) -> Dict[str, Any]:
        """
        Format single value results
        
        Args:
            result_value: Single value result
            query: Original query
            
        Returns:
            Formatted response
        """
        # Try to identify the type of result and format accordingly
        query_lower = query.lower()
        
        if isinstance(result_value, (int, float)):
            # Numeric result - likely an amount or count
            if "amount" in query_lower or "spent" in query_lower or "cost" in query_lower:
                return {
                    "response": f"The total amount is ${result_value:,.2f}",
                    "result_type": "amount",
                    "numeric_value": result_value,
                    "formatted_value": f"${result_value:,.2f}"
                }
            elif "count" in query_lower or "how many" in query_lower:
                return {
                    "response": f"There are {result_value:,} transactions",
                    "result_type": "count",
                    "numeric_value": result_value,
                    "formatted_value": f"{result_value:,}"
                }
            else:
                return {
                    "response": f"The result is {result_value}",
                    "result_type": "numeric",
                    "numeric_value": result_value,
                    "formatted_value": str(result_value)
                }
        
        elif isinstance(result_value, str):
            # String result - likely a category, vendor, or time period
            if any(word in query_lower for word in ["month", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
                return {
                    "response": f"The most expensive month is {result_value}",
                    "result_type": "month",
                    "string_value": result_value
                }
            elif any(word in query_lower for word in ["quarter", "q1", "q2", "q3", "q4"]):
                return {
                    "response": f"The most expensive quarter is {result_value}",
                    "result_type": "quarter",
                    "string_value": result_value
                }
            elif any(word in query_lower for word in ["vendor", "merchant", "store", "shop"]):
                return {
                    "response": f"The vendor you spent the most with is {result_value}",
                    "result_type": "vendor",
                    "string_value": result_value
                }
            elif any(word in query_lower for word in ["category", "type"]):
                return {
                    "response": f"The category with highest spending is {result_value}",
                    "result_type": "category",
                    "string_value": result_value
                }
            else:
                return {
                    "response": f"The result is {result_value}",
                    "result_type": "string",
                    "string_value": result_value
                }
        
        else:
            return {
                "response": f"The result is {result_value}",
                "result_type": "other",
                "raw_value": str(result_value)
            }
    
    def _format_series_result(self, result_value: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Format series results (dictionaries)
        
        Args:
            result_value: Series result as dictionary
            query: Original query
            
        Returns:
            Formatted response
        """
        if not result_value:
            return {
                "response": "No data found for your query",
                "result_type": "empty_series"
            }
        
        # Find the maximum value
        max_key = max(result_value.keys(), key=lambda k: result_value[k])
        max_value = result_value[max_key]
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["month", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
            month_name = self.month_names.get(int(max_key), max_key)
            return {
                "response": f"The most expensive month is {month_name} with ${max_value:,.2f}",
                "result_type": "monthly_series",
                "top_month": month_name,
                "top_amount": max_value,
                "all_data": result_value
            }
        
        elif any(word in query_lower for word in ["quarter", "q1", "q2", "q3", "q4"]):
            quarter_name = self.quarter_names.get(int(max_key), max_key)
            return {
                "response": f"The most expensive quarter is {quarter_name} with ${max_value:,.2f}",
                "result_type": "quarterly_series",
                "top_quarter": quarter_name,
                "top_amount": max_value,
                "all_data": result_value
            }
        
        else:
            return {
                "response": f"The highest value is {max_key} with ${max_value:,.2f}",
                "result_type": "generic_series",
                "top_key": max_key,
                "top_value": max_value,
                "all_data": result_value
            }
    
    def _format_dataframe_result(self, result_value: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Format dataframe results (list of dictionaries)
        
        Args:
            result_value: DataFrame result as list of dictionaries
            query: Original query
            
        Returns:
            Formatted response
        """
        if not result_value:
            return {
                "response": "No data found for your query",
                "result_type": "empty_dataframe",
                "table": [],
                "columns": [],
                "summary": "No data found."
            }
        
        query_lower = query.lower()
        columns = list(result_value[0].keys())
        n_rows = len(result_value)
        
        # Detect 'top N' queries
        import re
        top_n_match = re.search(r"top\s*(\d+)", query_lower)
        if top_n_match:
            n = int(top_n_match.group(1))
            summary = f"Top {n} results for your query."
        else:
            summary = f"Found {n_rows} results for your query."
        
        # Detect groupby queries (e.g., sum by month)
        groupby_keywords = ["by month", "by category", "by vendor", "by quarter", "group by", "monthly", "quarterly"]
        is_groupby = any(kw in query_lower for kw in groupby_keywords)
        
        # Prepare structured output
        table = result_value
        
        # Optionally, sort by amount descending for 'top N' queries
        if top_n_match and "amount" in columns:
            table = sorted(result_value, key=lambda x: x.get("amount", 0), reverse=True)[:n]
        
        # For groupby, sort by group key if present
        if is_groupby:
            group_keys = [col for col in columns if col not in ("amount", "sum", "total")]
            if group_keys:
                table = sorted(result_value, key=lambda x: tuple(x.get(k) for k in group_keys))
        
        # Build structured response
        response = {
            "response": summary,
            "result_type": "dataframe",
            "row_count": len(table),
            "columns": columns,
            "table": table,
            "summary": summary
        }
        
        # Add a sample text summary for top N
        if top_n_match and table:
            lines = []
            for i, row in enumerate(table, 1):
                desc = row.get("description") or row.get("vendor") or row.get("category") or ""
                date = row.get("date") or row.get("month") or row.get("quarter") or ""
                amount = row.get("amount") or row.get("sum") or row.get("total") or ""
                lines.append(f"{i}. ${amount:,.2f} on {date} at {desc}" if amount and date else str(row))
            response["text_list"] = lines
        
        # For groupby, add a text summary
        if is_groupby and table:
            lines = []
            for row in table:
                group_val = " - ".join(str(row.get(col, "")) for col in columns if col not in ("amount", "sum", "total"))
                amount = row.get("amount") or row.get("sum") or row.get("total") or ""
                lines.append(f"{group_val}: ${amount:,.2f}" if amount else str(row))
            response["text_list"] = lines
        
        return response
    
    def _format_scalar_result(self, result_value: Any, query: str) -> Dict[str, Any]:
        """
        Format scalar results
        
        Args:
            result_value: Scalar result
            query: Original query
            
        Returns:
            Formatted response
        """
        return {
            "response": f"The result is {result_value}",
            "result_type": "scalar",
            "value": result_value
        }
    
    def _format_unknown_result(self, result_value: Any, query: str) -> Dict[str, Any]:
        """
        Format unknown result types
        
        Args:
            result_value: Unknown result
            query: Original query
            
        Returns:
            Formatted response
        """
        return {
            "response": f"Analysis completed. Result: {result_value}",
            "result_type": "unknown",
            "raw_value": str(result_value)
        }
    
    def _format_error_response(self, execution_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Format error responses
        
        Args:
            execution_result: Execution result with error
            query: Original query
            
        Returns:
            Formatted error response
        """
        error_message = execution_result.get("error", "Unknown error occurred")
        
        return {
            "response": f"I encountered an error while analyzing your data: {error_message}. Please try rephrasing your question.",
            "result_type": "error",
            "error": error_message,
            "original_query": query,
            "success": False
        }
    
    def generate_follow_up_questions(self, formatted_result: Dict[str, Any], query: str) -> List[str]:
        """
        Generate follow-up questions based on the result
        
        Args:
            formatted_result: Formatted result
            query: Original query
            
        Returns:
            List of follow-up questions
        """
        follow_ups = []
        result_type = formatted_result.get("result_type", "")
        query_lower = query.lower()
        
        # Generate questions based on result type
        if result_type == "monthly_series":
            follow_ups.extend([
                "What was the least expensive month?",
                "How does this compare to last year?",
                "What caused the high spending in that month?"
            ])
        
        elif result_type == "quarterly_series":
            follow_ups.extend([
                "What was the least expensive quarter?",
                "How does this compare to last year?",
                "What caused the high spending in that quarter?"
            ])
        
        elif result_type == "vendor":
            follow_ups.extend([
                "What categories do I spend the most on?",
                "How much did I spend with that vendor?",
                "What other vendors do I spend a lot with?"
            ])
        
        elif result_type == "category":
            follow_ups.extend([
                "How much did I spend in that category?",
                "What vendors do I use most for that category?",
                "How does this compare to other categories?"
            ])
        
        else:
            # Generic follow-up questions
            follow_ups.extend([
                "Can you show me the breakdown by month?",
                "What are my top spending categories?",
                "How does this compare to previous periods?"
            ])
        
        return follow_ups[:3]  # Return max 3 questions
    
    def format_execution_summary(self, execution_history: List[Dict[str, Any]]) -> str:
        """
        Format execution history into readable summary
        
        Args:
            execution_history: List of execution steps
            
        Returns:
            Formatted summary string
        """
        if not execution_history:
            return "No operations were performed."
        
        successful_ops = [op for op in execution_history if op.get('success', False)]
        failed_ops = [op for op in execution_history if not op.get('success', False)]
        
        summary_parts = []
        summary_parts.append(f"Executed {len(execution_history)} operations")
        
        if successful_ops:
            summary_parts.append(f"({len(successful_ops)} successful)")
        
        if failed_ops:
            summary_parts.append(f"({len(failed_ops)} failed)")
        
        return " ".join(summary_parts) 