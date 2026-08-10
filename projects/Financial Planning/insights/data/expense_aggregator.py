"""
Expense Aggregator for deterministic category-based expense analysis
Provides monthly average expenses by category for external projects
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.google_sheets_connector import GoogleSheetsConnector
from data.data_processor import DataProcessor
from config.settings import DATA_SCHEMA

logger = logging.getLogger(__name__)


class ExpenseAggregator:
    """
    Deterministic expense aggregator for external projects
    Calculates monthly average expenses by category without using LLM
    """
    
    def __init__(self, 
                 spreadsheet_id: Optional[str] = None,
                 credentials_file: Optional[str] = None):
        """
        Initialize the expense aggregator
        
        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            credentials_file: Path to Google service account credentials file
        """
        self.sheets_connector = GoogleSheetsConnector(
            spreadsheet_id=spreadsheet_id,
            credentials_file=credentials_file
        )
        self.data_processor = DataProcessor()
        
    def get_monthly_average_expenses(self, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   years: Optional[int] = None) -> pd.DataFrame:
        """
        Get monthly average expenses by category for a given time period
        
        Args:
            start_date: Start date for analysis (overrides years)
            end_date: End date for analysis (overrides years)
            years: Number of years to look back (default: 2 years)
            
        Returns:
            DataFrame with columns: Category, Actual (monthly average)
        """
        try:
            # Determine date range
            if start_date and end_date:
                date_range = (start_date, end_date)
            else:
                if years is None:
                    years = 2  # Default to 2 years
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * years)
                date_range = (start_date, end_date)
            
            logger.info(f"Analyzing expenses from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
            
            # Get transaction data from Google Sheets
            tabs_data = self.sheets_connector.get_valid_transaction_data(
                required_columns=DATA_SCHEMA["required_columns"]
            )
            
            if not tabs_data:
                logger.warning("No valid transaction data found")
                return pd.DataFrame(columns=['Category', 'Actual'])
            
            # Process and clean the data
            processed_data = self.data_processor.process_transaction_data(tabs_data)
            
            # Filter by date range
            filtered_data = self._filter_by_date_range(processed_data, date_range[0], date_range[1])
            
            if filtered_data.empty:
                logger.warning("No data found for the specified date range")
                return pd.DataFrame(columns=['Category', 'Actual'])
            
            # Calculate monthly averages by category
            monthly_averages = self._calculate_monthly_averages(filtered_data, date_range)
            
            logger.info(f"Calculated monthly averages for {len(monthly_averages)} categories")
            return monthly_averages
            
        except Exception as e:
            logger.error(f"Error calculating monthly average expenses: {e}")
            return pd.DataFrame(columns=['Category', 'Actual'])
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Filter DataFrame by date range
        
        Args:
            df: DataFrame with transaction data
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered DataFrame
        """
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range
            filtered_df = df[
                (df['date'] >= start_date) & 
                (df['date'] <= end_date)
            ]
            
            logger.info(f"Filtered {len(filtered_df)} transactions from {len(df)} total")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering by date range: {e}")
            return df
    
    def _calculate_monthly_averages(self, df: pd.DataFrame, date_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """
        Calculate monthly average expenses by category
        
        Args:
            df: DataFrame with transaction data
            date_range: Tuple of (start_date, end_date)
            
        Returns:
            DataFrame with monthly average expenses by category
        """
        try:
            start_date, end_date = date_range
            
            # Calculate number of months in the period
            months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if months_diff == 0:
                months_diff = 1  # At least 1 month
            
            # Group by category and sum amounts
            category_totals = df.groupby('category')['amount'].sum()
            
            # Calculate monthly averages
            monthly_averages = category_totals / months_diff
            
            # Convert to DataFrame
            result_df = monthly_averages.reset_index()
            result_df.columns = ['Category', 'Actual']
            
            # Sort by amount (descending)
            result_df = result_df.sort_values('Actual', ascending=False)
            
            logger.info(f"Calculated monthly averages over {months_diff} months")
            logger.info(f"Total monthly expenses: ${result_df['Actual'].sum():,.2f}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating monthly averages: {e}")
            return pd.DataFrame(columns=['Category', 'Actual'])
    
    def get_expense_summary(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          years: Optional[int] = None) -> Dict[str, any]:
        """
        Get a summary of expense analysis
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            years: Number of years to analyze
            
        Returns:
            Dictionary with expense summary
        """
        try:
            expenses_df = self.get_monthly_average_expenses(
                start_date=start_date,
                end_date=end_date,
                years=years
            )
            
            if expenses_df.empty:
                return {
                    'total_monthly_expenses': 0,
                    'category_count': 0,
                    'period_months': 0,
                    'success': False
                }
            
            # Calculate period info
            if start_date and end_date:
                months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                if months_diff == 0:
                    months_diff = 1
            else:
                if years is None:
                    years = 2
                months_diff = years * 12
            
            return {
                'total_monthly_expenses': expenses_df['Actual'].sum(),
                'category_count': len(expenses_df),
                'highest_category': expenses_df.iloc[0]['Category'],
                'highest_amount': expenses_df.iloc[0]['Actual'],
                'lowest_category': expenses_df.iloc[-1]['Category'],
                'lowest_amount': expenses_df.iloc[-1]['Actual'],
                'period_months': months_diff,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error getting expense summary: {e}")
            return {
                'total_monthly_expenses': 0,
                'category_count': 0,
                'period_months': 0,
                'success': False,
                'error': str(e)
            } 