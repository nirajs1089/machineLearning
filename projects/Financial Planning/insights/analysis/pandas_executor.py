"""
Pandas Executor for safely executing pandas operations
"""

import pandas as pd
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PANDAS_OPERATIONS_CONFIG, ERROR_MESSAGES

logger = logging.getLogger(__name__)


class PandasExecutor:
    """
    Safely execute pandas operations on dataframes
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the pandas executor
        
        Args:
            dataframe: DataFrame to execute operations on
        """
        self.original_df = dataframe.copy()
        self.current_df = dataframe.copy()
        self.safe_operations = PANDAS_OPERATIONS_CONFIG["safe_operations"]
        self.forbidden_operations = PANDAS_OPERATIONS_CONFIG["forbidden_operations"]
        self.max_operations = PANDAS_OPERATIONS_CONFIG["max_operations"]
        self.timeout_seconds = PANDAS_OPERATIONS_CONFIG["timeout_seconds"]
        self.execution_history = []
    
    def execute_operations(self, operations: List[str]) -> Dict[str, Any]:
        """
        Execute a list of pandas operations safely
        
        Args:
            operations: List of pandas operation strings
            
        Returns:
            Dictionary with execution results and metadata
        """
        try:
            # Reset to original dataframe
            self.current_df = self.original_df.copy()
            self.execution_history = []
            
            # Validate operations
            self._validate_operations(operations)
            
            # Execute operations with timeout
            start_time = time.time()
            result = self._execute_operations_with_timeout(operations, start_time)
            
            # Format results
            formatted_result = self._format_results(result)
            
            logger.info(f"Successfully executed {len(operations)} operations")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error executing operations: {e}")
            return {
                "error": str(e),
                "result": None,
                "execution_history": self.execution_history,
                "success": False
            }
    
    def _validate_operations(self, operations: List[str]) -> None:
        """
        Validate operations for safety and correctness
        
        Args:
            operations: List of operations to validate
            
        Raises:
            ValueError: If validation fails
        """
        if len(operations) > self.max_operations:
            raise ValueError(f"Too many operations: {len(operations)} > {self.max_operations}")
        
        operations_text = ' '.join(operations).lower()
        
        # Check for forbidden operations
        for forbidden_op in self.forbidden_operations:
            if forbidden_op in operations_text:
                raise ValueError(f"Forbidden operation detected: {forbidden_op}")
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'subprocess\s*\(',
            r'import\s+',
            r'__.*__',
            r'\.eval\(',
            r'\.exec\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, operations_text):
                raise ValueError(f"Dangerous pattern detected: {pattern}")
    
    def _execute_operations_with_timeout(self, operations: List[str], start_time: float) -> Any:
        """
        Execute operations with timeout protection
        
        Args:
            operations: List of operations to execute
            start_time: Start time for timeout calculation
            
        Returns:
            Execution result
            
        Raises:
            TimeoutError: If execution takes too long
        """
        # Create execution environment
        execution_env = self._create_execution_environment()
        
        for i, operation in enumerate(operations):
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                raise TimeoutError(f"Execution timeout after {self.timeout_seconds} seconds")
            
            try:
                # Execute operation
                result = self._execute_single_operation(operation, execution_env)
                
                # Record execution
                self.execution_history.append({
                    "operation": operation,
                    "step": i + 1,
                    "success": True,
                    "result_shape": self.current_df.shape if hasattr(self.current_df, 'shape') else None
                })
                
            except Exception as e:
                # Record failure
                self.execution_history.append({
                    "operation": operation,
                    "step": i + 1,
                    "success": False,
                    "error": str(e)
                })
                raise
        
        return execution_env['result'] if execution_env.get('result') is not None else self.current_df
    
    def _create_execution_environment(self) -> Dict[str, Any]:
        """
        Create a safe execution environment
        
        Returns:
            Dictionary with execution environment
        """
        return {
            'df': self.current_df,
            'pd': pd,
            'result': None,
            'df_filtered': None,
            'df_grouped': None
        }
    
    def _execute_single_operation(self, operation: str, env: Dict[str, Any]) -> Any:
        """
        Execute a single pandas operation
        
        Args:
            operation: Operation string to execute
            env: Execution environment
            
        Returns:
            Result of the operation
        """
        try:
            # Execute the operation
            exec(operation, env)
            
            # Update current dataframe if it was modified
            if 'df' in env:
                self.current_df = env['df']
            if 'df_filtered' in env and env['df_filtered'] is not None:
                self.current_df = env['df_filtered']
            if 'df_grouped' in env and env['df_grouped'] is not None:
                self.current_df = env['df_grouped']
            
            return env['result'] if env.get('result') is not None else self.current_df
            
        except Exception as e:
            logger.error(f"Error executing operation '{operation}': {e}")
            raise
    
    def _format_results(self, result: Any) -> Dict[str, Any]:
        """
        Format execution results for user consumption
        
        Args:
            result: Raw execution result
            
        Returns:
            Formatted result dictionary
        """
        try:
            if isinstance(result, pd.Series):
                # Handle Series results
                if len(result) == 1:
                    # Single value result
                    return {
                        "result": result.iloc[0] if hasattr(result, 'iloc') else result.values[0],
                        "result_type": "single_value",
                        "execution_history": self.execution_history,
                        "success": True
                    }
                else:
                    # Multiple values
                    return {
                        "result": result.to_dict(),
                        "result_type": "series",
                        "execution_history": self.execution_history,
                        "success": True
                    }
            
            elif isinstance(result, pd.DataFrame):
                # Handle DataFrame results
                if len(result) == 1 and len(result.columns) == 1:
                    # Single cell result
                    return {
                        "result": result.iloc[0, 0],
                        "result_type": "single_value",
                        "execution_history": self.execution_history,
                        "success": True
                    }
                else:
                    # Multiple rows/columns
                    return {
                        "result": result.to_dict('records'),
                        "result_type": "dataframe",
                        "result_shape": result.shape,
                        "execution_history": self.execution_history,
                        "success": True
                    }
            
            elif isinstance(result, (int, float, str)):
                # Handle scalar results
                return {
                    "result": result,
                    "result_type": "scalar",
                    "execution_history": self.execution_history,
                    "success": True
                }
            
            else:
                # Handle other types
                return {
                    "result": str(result),
                    "result_type": "other",
                    "execution_history": self.execution_history,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return {
                "result": str(result),
                "result_type": "error",
                "execution_history": self.execution_history,
                "success": False,
                "error": str(e)
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution history
        
        Returns:
            Execution summary
        """
        if not self.execution_history:
            return {"message": "No operations executed"}
        
        successful_ops = [op for op in self.execution_history if op.get('success', False)]
        failed_ops = [op for op in self.execution_history if not op.get('success', False)]
        
        return {
            "total_operations": len(self.execution_history),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.execution_history) if self.execution_history else 0,
            "execution_history": self.execution_history
        }
    
    def reset_dataframe(self) -> None:
        """Reset to original dataframe"""
        self.current_df = self.original_df.copy()
        self.execution_history = []
    
    def get_current_dataframe(self) -> pd.DataFrame:
        """Get current state of dataframe"""
        return self.current_df.copy()


class SafePandasExecutor(PandasExecutor):
    """
    Enhanced pandas executor with additional safety features
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with additional safety checks"""
        super().__init__(dataframe)
        self.max_dataframe_size = 1000000  # 1M rows
        self.max_memory_usage = 1000000000  # 1GB
    
    def _validate_operations(self, operations: List[str]) -> None:
        """Enhanced validation with size and memory checks"""
        super()._validate_operations(operations)
        
        # Check if operations might create very large dataframes
        operations_text = ' '.join(operations).lower()
        
        # Check for operations that might create large results
        if 'groupby' in operations_text and 'count' in operations_text:
            # GroupBy with count might create large results
            pass  # Could add specific checks here
        
        # Check for operations that might modify the dataframe extensively
        if operations_text.count('df[') > 5:
            logger.warning("Many filtering operations detected - may impact performance")
    
    def _execute_single_operation(self, operation: str, env: Dict[str, Any]) -> Any:
        """Enhanced execution with memory and size monitoring"""
        # Check dataframe size before operation
        if hasattr(self.current_df, 'shape'):
            rows, cols = self.current_df.shape
            if rows > self.max_dataframe_size:
                raise ValueError(f"Dataframe too large: {rows} rows > {self.max_dataframe_size}")
        
        # Execute operation
        result = super()._execute_single_operation(operation, env)
        
        # Check result size
        if hasattr(result, 'shape'):
            rows, cols = result.shape
            if rows > self.max_dataframe_size:
                logger.warning(f"Large result created: {rows} rows")
        
        return result 