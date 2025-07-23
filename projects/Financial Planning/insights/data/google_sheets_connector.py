"""
Google Sheets Connector for fetching transaction data from multiple tabs
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GOOGLE_SHEETS_CONFIG, ERROR_MESSAGES

logger = logging.getLogger(__name__)


class GoogleSheetsConnector:
    """
    Connector class for fetching transaction data from Google Sheets
    """
    
    def __init__(self, 
                 spreadsheet_id: Optional[str] = None, 
                 credentials_file: Optional[str] = None,
                 tab_names: Optional[List[str]] = None,
                 data_range: Optional[str] = None):
        """
        Initialize the Google Sheets connector
        
        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            credentials_file: Path to Google service account credentials file
            tab_names: List of tab names to fetch data from
            data_range: Range specification for data (e.g., 'A:D')
        """
        self.spreadsheet_id = spreadsheet_id or GOOGLE_SHEETS_CONFIG["spreadsheet_id"]
        self.credentials_file = credentials_file or GOOGLE_SHEETS_CONFIG["credentials_file"]
        self.scopes = GOOGLE_SHEETS_CONFIG["scopes"]
        self.tab_names = tab_names or GOOGLE_SHEETS_CONFIG["tab_names"]
        self.data_range = data_range or GOOGLE_SHEETS_CONFIG["data_range"]
        self.service = None
        self._authenticate()
    
    def _authenticate(self) -> None:
        """
        Authenticate with Google Sheets API
        """
        try:
            if not os.path.exists(self.credentials_file):
                raise FileNotFoundError(f"Credentials file not found: {self.credentials_file}")
            
            credentials = Credentials.from_service_account_file(
                self.credentials_file, 
                scopes=self.scopes
            )
            
            self.service = build('sheets', 'v4', credentials=credentials)
            logger.info("Successfully authenticated with Google Sheets API")
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise Exception(ERROR_MESSAGES["missing_credentials"])
    
    def get_sheet_names(self) -> List[str]:
        """
        Get all sheet names from the spreadsheet
        
        Returns:
            List of sheet names
        """
        try:
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            
            sheet_names = [sheet['properties']['title'] for sheet in spreadsheet['sheets']]
            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            return sheet_names
            
        except HttpError as e:
            logger.error(f"Error fetching sheet names: {e}")
            raise Exception(ERROR_MESSAGES["invalid_spreadsheet"])
    
    def get_sheet_data(self, sheet_name: str, range_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get data from a specific sheet
        
        Args:
            sheet_name: Name of the sheet to fetch data from
            range_name: Optional range specification (e.g., 'A:D')
            
        Returns:
            DataFrame containing the sheet data
        """
        try:
            # Use configured range if not specified
            if not range_name:
                range_name = f"{sheet_name}!{self.data_range}"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                logger.warning(f"No data found in sheet: {sheet_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(values[1:], columns=values[0])
            
            # Add tab name as a column
            df['tab_name'] = sheet_name
            
            logger.info(f"Fetched {len(df)} rows from sheet: {sheet_name}")
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching data from sheet {sheet_name}: {e}")
            raise Exception(f"Unable to fetch data from sheet: {sheet_name}")
    
    def get_transaction_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get transaction data from specified tabs
        
        Returns:
            Dictionary mapping tab names to DataFrames
        """
        try:
            all_data = {}
            
            for tab_name in self.tab_names:
                logger.info(f"Fetching data from tab: {tab_name}")
                df = self.get_sheet_data(tab_name)
                
                if not df.empty:
                    all_data[tab_name] = df
                else:
                    logger.warning(f"Skipping empty tab: {tab_name}")
            
            logger.info(f"Successfully fetched data from {len(all_data)} tabs")
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching transaction data: {e}")
            raise
    
    def validate_sheet_structure(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that a sheet has the required columns for transaction data
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        if df.empty:
            return False
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def get_valid_transaction_data(self, required_columns: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get only tabs that contain valid transaction data
        
        Args:
            required_columns: List of required columns for transaction data
            
        Returns:
            Dictionary of valid transaction tabs
        """
        all_data = self.get_transaction_data()
        valid_data = {}
        
        for tab_name, df in all_data.items():
            if self.validate_sheet_structure(df, required_columns):
                valid_data[tab_name] = df
                logger.info(f"Validated transaction tab: {tab_name}")
            else:
                logger.warning(f"Skipping invalid transaction tab: {tab_name}")
        
        if not valid_data:
            raise Exception(ERROR_MESSAGES["missing_columns"])
        
        logger.info(f"Found {len(valid_data)} valid transaction tabs")
        return valid_data
    
    def test_connection(self) -> bool:
        """
        Test the connection to Google Sheets
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            sheet_names = self.get_sheet_names()
            logger.info("Google Sheets connection test successful")
            return True
        except Exception as e:
            logger.error(f"Google Sheets connection test failed: {e}")
            return False


class MockGoogleSheetsConnector:
    """
    Mock connector for testing purposes
    """
    
    def __init__(self, test_data: Optional[Dict[str, pd.DataFrame]] = None):
        self.test_data = test_data or self._create_test_data()
        self.tab_names = ["Citi", "Chase", "Cash"]
    
    def _create_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create mock test data"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample transaction data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        descriptions = ['Amazon Purchase', 'Starbucks Coffee', 'Uber Ride', 'Netflix Subscription', 'Whole Foods', 'Shell Gas', 'Target Shopping']
        categories = ['Shopping', 'Food', 'Transportation', 'Entertainment', 'Food', 'Transportation', 'Shopping']
        
        transactions = []
        for i in range(100):
            date = np.random.choice(dates)
            description = np.random.choice(descriptions)
            amount = round(np.random.uniform(10, 500), 2)
            category = np.random.choice(categories)
            
            transactions.append({
                'date': date.strftime('%Y-%m-%d'),
                'description': description,
                'amount': amount,
                'category': category
            })
        
        df = pd.DataFrame(transactions)
        
        return {
            'Citi': df.copy(),
            'Chase': df.copy(),
            'Cash': df.copy()
        }
    
    def get_sheet_names(self) -> List[str]:
        return self.tab_names
    
    def get_sheet_data(self, sheet_name: str, range_name: Optional[str] = None) -> pd.DataFrame:
        df = self.test_data.get(sheet_name, pd.DataFrame())
        if not df.empty:
            df['tab_name'] = sheet_name
        return df
    
    def get_transaction_data(self) -> Dict[str, pd.DataFrame]:
        return self.test_data
    
    def get_valid_transaction_data(self, required_columns: List[str]) -> Dict[str, pd.DataFrame]:
        return self.test_data
    
    def test_connection(self) -> bool:
        return True 