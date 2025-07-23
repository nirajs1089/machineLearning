"""
Data Processor for cleaning and preparing transaction data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import re
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_SCHEMA, ERROR_MESSAGES

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Process and clean transaction data from multiple sources
    """
    
    def __init__(self, date_format: str = None):
        """
        Initialize the data processor
        
        Args:
            date_format: Expected date format in the data
        """
        self.date_format = date_format or DATA_SCHEMA["date_format"]
        self.required_columns = DATA_SCHEMA["required_columns"]
        self.amount_column = DATA_SCHEMA["amount_column"]
        self.category_column = DATA_SCHEMA["category_column"]
        self.description_column = DATA_SCHEMA["description_column"]
        self.date_column = DATA_SCHEMA["date_column"]
    
    def combine_tabs_data(self, tabs_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from multiple tabs into a single DataFrame
        
        Args:
            tabs_data: Dictionary mapping tab names to DataFrames
            
        Returns:
            Combined DataFrame with tab name information
        """
        combined_data = []
        
        for tab_name, df in tabs_data.items():
            if not df.empty:
                # Add tab name if not already present
                if 'tab_name' not in df.columns:
                    df_copy = df.copy()
                    df_copy['tab_name'] = tab_name
                    combined_data.append(df_copy)
                else:
                    combined_data.append(df)
        
        if not combined_data:
            raise Exception(ERROR_MESSAGES["no_data_found"])
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Combined data from {len(combined_data)} tabs: {len(combined_df)} total rows")
        
        return combined_df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with cleaned column names
        """
        df_clean = df.copy()
        
        # Remove extra whitespace and convert to lowercase
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        
        # Map common variations to standard names
        column_mapping = {
            'transaction_date': 'date',
            'post_date': 'date',
            'merchant': 'description',
            'vendor': 'description',
            'transaction_amount': 'amount',
            'spend_amount': 'amount',
            'transaction_category': 'category',
            'spend_category': 'category',
            'transaction_description': 'description',
            'merchant_name': 'description'
        }
        
        df_clean = df_clean.rename(columns=column_mapping)
        
        logger.info(f"Cleaned column names: {list(df_clean.columns)}")
        return df_clean
    
    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns are present
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all required columns are present
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("All required columns are present")
        return True
    
    def clean_date_column(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Clean and standardize date column
        
        Args:
            df: DataFrame to clean
            date_column: Name of the date column
            
        Returns:
            DataFrame with cleaned date column
        """
        df_clean = df.copy()
        
        if date_column not in df_clean.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return df_clean
        
        # Convert to datetime, handling various formats
        try:
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column], 
                format=self.date_format,
                errors='coerce'
            )
            
            # Handle rows where date conversion failed
            invalid_dates = df_clean[date_column].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} rows with invalid dates")
                df_clean = df_clean.dropna(subset=[date_column])
            
            logger.info(f"Cleaned date column: {len(df_clean)} valid rows")
            
        except Exception as e:
            logger.error(f"Error cleaning date column: {e}")
            raise Exception(ERROR_MESSAGES["invalid_date_format"])
        
        return df_clean
    
    def clean_amount_column(self, df: pd.DataFrame, amount_column: str = 'amount') -> pd.DataFrame:
        """
        Clean and standardize amount column
        
        Args:
            df: DataFrame to clean
            amount_column: Name of the amount column
            
        Returns:
            DataFrame with cleaned amount column
        """
        df_clean = df.copy()
        
        if amount_column not in df_clean.columns:
            logger.warning(f"Amount column '{amount_column}' not found")
            return df_clean
        
        # Remove currency symbols and convert to numeric
        df_clean[amount_column] = df_clean[amount_column].astype(str).str.replace(
            r'[^\d.-]', '', regex=True
        )
        
        # Convert to float
        df_clean[amount_column] = pd.to_numeric(df_clean[amount_column], errors='coerce')
        
        # Remove rows with invalid amounts
        invalid_amounts = df_clean[amount_column].isna().sum()
        if invalid_amounts > 0:
            logger.warning(f"Found {invalid_amounts} rows with invalid amounts")
            df_clean = df_clean.dropna(subset=[amount_column])
        
        logger.info(f"Cleaned amount column: {len(df_clean)} valid rows")
        return df_clean
    
    def clean_category_column(self, df: pd.DataFrame, category_column: str = 'category') -> pd.DataFrame:
        """
        Clean and standardize category column
        
        Args:
            df: DataFrame to clean
            category_column: Name of the category column
            
        Returns:
            DataFrame with cleaned category column
        """
        df_clean = df.copy()
        
        if category_column not in df_clean.columns:
            logger.warning(f"Category column '{category_column}' not found")
            return df_clean
        
        # Clean category names
        df_clean[category_column] = df_clean[category_column].astype(str).str.strip().str.title()
        
        # Map common variations to standard categories
        category_mapping = {
            'Food & Drink': 'Food',
            'Food NV': 'Food',
            'Restaurants': 'Food',
            'Dining': 'Food',
            'Transport': 'Travel',
            'Gas': 'Travel',
            'Fuel': 'Travel',
            'Entertainment': 'Shopping',
            'Movies': 'Shopping',
            'Shopping': 'Shopping',
            'Retail': 'Shopping',
            'Education': 'Education',
            'Healthcare': 'Health',
            'Medical': 'Health',
            'Travel': 'Travel',
            'Vacation': 'Vacation',
            'Business': 'Business',
            'Work': 'Business',
            'Gifts': 'Gifts',
            'Gift': 'Gifts',
            'gifts': 'Gifts'
        }
        
        df_clean[category_column] = df_clean[category_column].replace(category_mapping)
        
        logger.info(f"Cleaned category column: {df_clean[category_column].nunique()} unique categories")
        return df_clean
    
    def clean_description_column(self, df: pd.DataFrame, description_column: str = 'description') -> pd.DataFrame:
        """
        Clean and standardize description column
        
        Args:
            df: DataFrame to clean
            description_column: Name of the description column
            
        Returns:
            DataFrame with cleaned description column
        """
        df_clean = df.copy()
        
        if description_column not in df_clean.columns:
            logger.warning(f"Description column '{description_column}' not found")
            return df_clean
        
        # Clean description names
        df_clean[description_column] = df_clean[description_column].astype(str).str.strip()
        
        # Remove common prefixes/suffixes
        df_clean[description_column] = df_clean[description_column].str.replace(
            r'^(AMZN|AMAZON\.COM|AMZN MKTP)', 'Amazon', regex=True, case=False
        )
        df_clean[description_column] = df_clean[description_column].str.replace(
            r'^(STARBUCKS|SBUX)', 'Starbucks', regex=True, case=False
        )
        
        logger.info(f"Cleaned description column: {df_clean[description_column].nunique()} unique descriptions")
        return df_clean
    
    def add_time_periods(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Add time period columns for analysis
        
        Args:
            df: DataFrame to process
            date_column: Name of the date column
            
        Returns:
            DataFrame with additional time period columns
        """
        df_processed = df.copy()
        
        if date_column not in df_processed.columns:
            logger.warning(f"Date column '{date_column}' not found, skipping time period addition")
            return df_processed
        
        # Add various time period columns
        df_processed['year'] = df_processed[date_column].dt.year
        df_processed['month'] = df_processed[date_column].dt.month
        df_processed['quarter'] = df_processed[date_column].dt.quarter
        df_processed['week'] = df_processed[date_column].dt.isocalendar().week
        df_processed['day_of_week'] = df_processed[date_column].dt.day_name()
        
        # Add quarter labels
        df_processed['quarter_label'] = 'Q' + df_processed['quarter'].astype(str)
        
        # Add month labels
        df_processed['month_label'] = df_processed[date_column].dt.strftime('%B')
        
        logger.info("Added time period columns for analysis")
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transactions
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df_clean = df.drop_duplicates()
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate transactions")
        
        return df_clean
    
    def filter_valid_transactions(self, df: pd.DataFrame, min_amount: float = -10000.01) -> pd.DataFrame:
        """
        Filter out invalid transactions
        
        Args:
            df: DataFrame to filter
            min_amount: Minimum valid transaction amount
            
        Returns:
            DataFrame with valid transactions only
        """
        df_filtered = df.copy()
        
        # Filter by amount
        if self.amount_column in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[self.amount_column] >= min_amount]
        
        # Filter by date (remove future dates)
        if self.date_column in df_filtered.columns:
            today = datetime.now()
            df_filtered = df_filtered[df_filtered[self.date_column] <= today]
        
        logger.info(f"Filtered to {len(df_filtered)} valid transactions")
        return df_filtered
    
    def process_transaction_data(self, tabs_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Complete data processing pipeline
        
        Args:
            tabs_data: Dictionary mapping tab names to DataFrames
            
        Returns:
            Processed and cleaned DataFrame
        """
        logger.info("Starting data processing pipeline")
        
        try:
            # Step 1: Combine data from all tabs
            combined_df = self.combine_tabs_data(tabs_data)
            
            # Step 2: Clean column names
            combined_df = self.clean_column_names(combined_df)
            
            # Step 3: Validate required columns
            if not self.validate_required_columns(combined_df):
                raise Exception(ERROR_MESSAGES["missing_columns"])
            
            # Step 4: Clean individual columns
            combined_df = self.clean_date_column(combined_df)
            combined_df = self.clean_amount_column(combined_df)
            combined_df = self.clean_category_column(combined_df)
            combined_df = self.clean_description_column(combined_df)
            
            # Step 5: Add time period columns
            combined_df = self.add_time_periods(combined_df)
            
            # Step 6: Remove duplicates
            # combined_df = self.remove_duplicates(combined_df)
            
            # Step 7: Filter valid transactions
            combined_df = self.filter_valid_transactions(combined_df)
            
            logger.info(f"Data processing completed: {len(combined_df)} final transactions")
            return combined_df
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the processed data
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {"error": "No data available"}
        
        summary = {
            "total_transactions": len(df),
            "date_range": {
                "start": df[self.date_column].min().strftime('%Y-%m-%d') if self.date_column in df.columns else None,
                "end": df[self.date_column].max().strftime('%Y-%m-%d') if self.date_column in df.columns else None
            },
            "total_amount": df[self.amount_column].sum() if self.amount_column in df.columns else 0,
            "average_amount": df[self.amount_column].mean() if self.amount_column in df.columns else 0,
            "categories": df[self.category_column].value_counts().to_dict() if self.category_column in df.columns else {},
            "descriptions": df[self.description_column].value_counts().head(10).to_dict() if self.description_column in df.columns else {},
            "tabs": df['tab_name'].value_counts().to_dict() if 'tab_name' in df.columns else {}
        }
        
        return summary 