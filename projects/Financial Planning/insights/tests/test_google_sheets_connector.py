"""
Test file for Google Sheets Connector
"""

import unittest
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.google_sheets_connector import GoogleSheetsConnector, MockGoogleSheetsConnector


class TestMockGoogleSheetsConnector(unittest.TestCase):
    """Test the mock Google Sheets connector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.connector = MockGoogleSheetsConnector()
    
    def test_get_sheet_names(self):
        """Test getting sheet names"""
        sheet_names = self.connector.get_sheet_names()
        expected_names = ["Citi", "Chase", "Cash"]
        
        self.assertEqual(sheet_names, expected_names)
        self.assertEqual(len(sheet_names), 3)
    
    def test_get_sheet_data(self):
        """Test getting data from a specific sheet"""
        # Test with valid sheet name
        df = self.connector.get_sheet_data("Citi")
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('tab_name', df.columns)
        self.assertEqual(df['tab_name'].iloc[0], "Citi")
        
        # Test required columns
        required_columns = ['date', 'description', 'amount', 'category']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_get_transaction_data(self):
        """Test getting transaction data from all tabs"""
        tabs_data = self.connector.get_transaction_data()
        
        self.assertIsInstance(tabs_data, dict)
        self.assertEqual(len(tabs_data), 3)
        
        for tab_name in ["Citi", "Chase", "Cash"]:
            self.assertIn(tab_name, tabs_data)
            df = tabs_data[tab_name]
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(df.empty)
    
    def test_get_valid_transaction_data(self):
        """Test getting valid transaction data"""
        required_columns = ['date', 'description', 'amount', 'category']
        valid_data = self.connector.get_valid_transaction_data(required_columns)
        
        self.assertIsInstance(valid_data, dict)
        self.assertEqual(len(valid_data), 3)
        
        for tab_name, df in valid_data.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(df.empty)
            
            # Check that all required columns are present
            for col in required_columns:
                self.assertIn(col, df.columns)
    
    def test_test_connection(self):
        """Test connection test"""
        result = self.connector.test_connection()
        self.assertTrue(result)


class TestGoogleSheetsConnector(unittest.TestCase):
    """Test the real Google Sheets connector with mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_credentials_file = "test_credentials.json"
        self.mock_spreadsheet_id = "test_spreadsheet_id"
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_initialization_success(self, mock_build, mock_credentials):
        """Test successful initialization"""
        # Mock the credentials and service
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            self.assertEqual(connector.spreadsheet_id, self.mock_spreadsheet_id)
            self.assertEqual(connector.credentials_file, self.mock_credentials_file)
            self.assertIsNotNone(connector.service)
    
    @patch('os.path.exists')
    def test_initialization_missing_credentials(self, mock_exists):
        """Test initialization with missing credentials file"""
        mock_exists.return_value = False
        
        with self.assertRaises(Exception):
            GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_get_sheet_names_success(self, mock_build, mock_credentials):
        """Test successful sheet names retrieval"""
        # Mock the service and response
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        mock_response = {
            'sheets': [
                {'properties': {'title': 'Citi'}},
                {'properties': {'title': 'Chase'}},
                {'properties': {'title': 'Cash'}}
            ]
        }
        mock_service.spreadsheets().get().execute.return_value = mock_response
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            sheet_names = connector.get_sheet_names()
            expected_names = ["Citi", "Chase", "Cash"]
            
            self.assertEqual(sheet_names, expected_names)
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_get_sheet_data_success(self, mock_build, mock_credentials):
        """Test successful sheet data retrieval"""
        # Mock the service and response
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        mock_response = {
            'values': [
                ['date', 'description', 'amount', 'category'],
                ['2024-01-01', 'Test Transaction', '100.00', 'Food'],
                ['2024-01-02', 'Another Transaction', '50.00', 'Transportation']
            ]
        }
        mock_service.spreadsheets().values().get().execute.return_value = mock_response
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            df = connector.get_sheet_data("Citi")
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)  # 2 data rows
            self.assertIn('tab_name', df.columns)
            self.assertEqual(df['tab_name'].iloc[0], "Citi")
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_get_sheet_data_empty(self, mock_build, mock_credentials):
        """Test sheet data retrieval with empty sheet"""
        # Mock the service and response
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        mock_response = {'values': []}
        mock_service.spreadsheets().values().get().execute.return_value = mock_response
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            df = connector.get_sheet_data("EmptySheet")
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(df.empty)
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_validate_sheet_structure_valid(self, mock_build, mock_credentials):
        """Test sheet structure validation with valid data"""
        # Mock the service
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            # Create test dataframe with required columns
            test_df = pd.DataFrame({
                'date': ['2024-01-01'],
                'description': ['Test'],
                'amount': [100.0],
                'category': ['Food']
            })
            
            required_columns = ['date', 'description', 'amount', 'category']
            result = connector.validate_sheet_structure(test_df, required_columns)
            
            self.assertTrue(result)
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_validate_sheet_structure_invalid(self, mock_build, mock_credentials):
        """Test sheet structure validation with invalid data"""
        # Mock the service
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            # Create test dataframe with missing columns
            test_df = pd.DataFrame({
                'date': ['2024-01-01'],
                'description': ['Test']
                # Missing 'amount' and 'category' columns
            })
            
            required_columns = ['date', 'description', 'amount', 'category']
            result = connector.validate_sheet_structure(test_df, required_columns)
            
            self.assertFalse(result)
    
    @patch('data.google_sheets_connector.Credentials')
    @patch('data.google_sheets_connector.build')
    def test_test_connection_success(self, mock_build, mock_credentials):
        """Test successful connection test"""
        # Mock the service and response
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        mock_response = {
            'sheets': [
                {'properties': {'title': 'Citi'}},
                {'properties': {'title': 'Chase'}},
                {'properties': {'title': 'Cash'}}
            ]
        }
        mock_service.spreadsheets().get().execute.return_value = mock_response
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            connector = GoogleSheetsConnector(
                spreadsheet_id=self.mock_spreadsheet_id,
                credentials_file=self.mock_credentials_file
            )
            
            result = connector.test_connection()
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main() 