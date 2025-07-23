"""
Test suite for the intelligent LangChain extract agent
"""

import unittest
import pandas as pd
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import shutil

# Add the parent directory to path for imports
sys.path.insert(0, "/Users/vishankagandhi/Documents/")

from cost_allocation.prod.extract.extract_agent_langchain import (
    LangChainExtractAgent,
    IntelligentBankDecisionTool,
    AdaptiveExtractionTool,
    CSVExtractionTool,
    PlaidExtractionTool,
    DataValidationTool,
    extract
)


class TestIntelligentBankDecisionTool(unittest.TestCase):
    """Test the intelligent bank decision tool"""
    
    def setUp(self):
        self.tool = IntelligentBankDecisionTool()
    
    @patch('cost_allocation.prod.extract.extract_agent_langchain.ChatOpenAI')
    def test_ai_decision_making(self, mock_llm):
        """Test AI-powered decision making"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
DECISION: csv_extraction
REASONING: Chase is known to provide reliable CSV exports and this appears to be a Chase bank
CONFIDENCE: high
ALTERNATIVES: plaid_extraction if CSV is not available
"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = self.tool._run("Chase Bank")
        
        self.assertIn("csv_extraction", result)
        self.assertIn("high", result)
        self.assertEqual(len(self.tool.decision_history), 1)
    
    def test_learning_from_feedback(self):
        """Test learning capabilities"""
        self.tool.learn_from_feedback("Chase", "csv_extraction", True, "Worked perfectly")
        self.tool.learn_from_feedback("Unknown Bank", "plaid_extraction", False, "API not supported")
        
        insights = self.tool.get_learning_insights()
        
        self.assertEqual(insights["total_attempts"], 2)
        self.assertEqual(insights["success_rate"], 0.5)
        self.assertIn("csv_extraction", insights["method_effectiveness"])
    
    def test_empty_learning_data(self):
        """Test handling of empty learning data"""
        insights = self.tool.get_learning_insights()
        self.assertIn("No learning data available", insights["message"])


class TestAdaptiveExtractionTool(unittest.TestCase):
    """Test the adaptive extraction tool"""
    
    def setUp(self):
        self.tool = AdaptiveExtractionTool()
    
    @patch('cost_allocation.prod.extract.extract_agent_langchain.ChatOpenAI')
    def test_successful_extraction(self, mock_llm):
        """Test successful extraction adaptation"""
        result = self.tool._run("Chase", "csv_extraction")
        
        self.assertIn("SUCCESS", result)
        self.assertEqual(len(self.tool.extraction_attempts), 1)
        self.assertTrue(self.tool.extraction_attempts[0]["success"])
    
    @patch('cost_allocation.prod.extract.extract_agent_langchain.ChatOpenAI')
    def test_failed_extraction_adaptation(self, mock_llm):
        """Test adaptation when extraction fails"""
        # Mock LLM response for adaptation
        mock_response = Mock()
        mock_response.content = """
RECOMMENDATION: plaid_extraction
REASONING: CSV extraction failed, trying Plaid API as alternative
PARAMETERS: Use interactive mode for better user experience
"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = self.tool._run("Unknown Bank", "csv_extraction", "No CSV files found")
        
        self.assertIn("ADAPTATION", result)
        self.assertIn("plaid_extraction", result)
        self.assertEqual(len(self.tool.extraction_attempts), 1)
        self.assertFalse(self.tool.extraction_attempts[0]["success"])
    
    def test_adaptation_insights(self):
        """Test adaptation insights"""
        # Add some test data
        self.tool.extraction_attempts = [
            {"bank_name": "Chase", "method": "csv_extraction", "success": True, "failure_reason": "", "timestamp": "2025-01-01"},
            {"bank_name": "Unknown", "method": "csv_extraction", "success": False, "failure_reason": "No files found", "timestamp": "2025-01-01"},
            {"bank_name": "Unknown", "method": "plaid_extraction", "success": True, "failure_reason": "", "timestamp": "2025-01-01"}
        ]
        
        insights = self.tool.get_adaptation_insights()
        
        self.assertEqual(insights["total_attempts"], 3)
        self.assertEqual(insights["success_rate"], 2/3)
        self.assertIn("No files found", insights["failure_patterns"])


class TestCSVExtractionTool(unittest.TestCase):
    """Test the CSV extraction tool"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tool = CSVExtractionTool(self.temp_dir)
        
        # Create a test CSV file
        self.test_csv_path = os.path.join(self.temp_dir, "test_chase_transactions.csv")
        test_data = pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-02"],
            "Description": ["Test Transaction 1", "Test Transaction 2"],
            "Amount": [100.00, -50.00]
        })
        test_data.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_successful_csv_extraction(self):
        """Test successful CSV extraction"""
        result = self.tool._run("chase", "*chase*.csv")
        
        self.assertIn("SUCCESS", result)
        self.assertIn("2 transactions", result)
    
    def test_csv_file_not_found(self):
        """Test handling when CSV file is not found"""
        result = self.tool._run("nonexistent", "*nonexistent*.csv")
        
        self.assertIn("ERROR", result)
        self.assertIn("No CSV files found", result)


class TestPlaidExtractionTool(unittest.TestCase):
    """Test the Plaid extraction tool"""
    
    def setUp(self):
        self.tool = PlaidExtractionTool()
    
    @patch.dict(os.environ, {'PLAID_CLIENT_ID': 'test_id', 'PLAID_SECRET': 'test_secret'})
    def test_plaid_credentials_configured(self):
        """Test when Plaid credentials are configured"""
        self.assertEqual(self.tool.plaid_client_id, 'test_id')
        self.assertEqual(self.tool.plaid_secret, 'test_secret')
    
    @patch.dict(os.environ, {}, clear=True)
    def test_plaid_credentials_missing(self):
        """Test when Plaid credentials are missing"""
        tool = PlaidExtractionTool()
        result = tool._run("test_bank")
        
        self.assertIn("ERROR", result)
        self.assertIn("not configured", result)


class TestDataValidationTool(unittest.TestCase):
    """Test the data validation tool"""
    
    def setUp(self):
        self.tool = DataValidationTool()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('cost_allocation.prod.extract.extract_agent_langchain.ChatOpenAI')
    def test_ai_data_validation(self, mock_llm):
        """Test AI-powered data validation"""
        # Create test CSV file
        test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        test_data = pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-02"],
            "Description": ["Test 1", "Test 2"],
            "Amount": [100.00, -50.00]
        })
        test_data.to_csv(test_csv_path, index=False)
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
QUALITY_SCORE: 9
COMPLETENESS: 100%
ISSUES: None found
RECOMMENDATIONS: Data looks good
"""
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = self.tool._run(test_csv_path)
        
        self.assertIn("VALIDATION PASSED", result)
        self.assertIn("AI ANALYSIS", result)
    
    def test_file_not_found(self):
        """Test validation when file is not found"""
        result = self.tool._run("/nonexistent/file.csv")
        
        self.assertIn("ERROR", result)
        self.assertIn("File not found", result)


class TestLangChainExtractAgent(unittest.TestCase):
    """Test the main LangChain extract agent"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.agent = LangChainExtractAgent(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.agent)
        self.assertEqual(len(self.agent.tools), 5)  # 5 intelligent tools
        self.assertIsNotNone(self.agent.llm)
    
    def test_extraction_history_tracking(self):
        """Test extraction history tracking"""
        # Simulate extraction attempt
        self.agent.extraction_history.append({
            "timestamp": datetime.now().isoformat(),
            "bank_name": "test_bank",
            "status": "started"
        })
        
        summary = self.agent.get_extraction_summary()
        
        self.assertEqual(summary["total_extractions"], 1)
        self.assertEqual(summary["successful_extractions"], 0)
    
    def test_intelligence_insights(self):
        """Test intelligence insights collection"""
        insights = self.agent.get_intelligence_insights()
        
        self.assertIn("bank_decisions", insights)
        self.assertIn("adaptation_patterns", insights)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing pipeline"""
    
    @patch('cost_allocation.prod.extract.extract_agent_langchain.LangChainExtractAgent')
    def test_extract_function(self, mock_agent_class):
        """Test the backward-compatible extract function"""
        # Mock the agent
        mock_agent = Mock()
        mock_agent.extract.return_value = (pd.DataFrame({"test": [1, 2, 3]}), "test_bank")
        mock_agent_class.return_value = mock_agent
        
        # Test the function
        df, bank_name = extract("test_bank", export_csv=False)
        
        # Verify the agent was called correctly
        mock_agent.extract.assert_called_once_with("test_bank", False)
        
        # Verify return values
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(bank_name, "test_bank")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        chase_csv = os.path.join(self.temp_dir, "chase_transactions.csv")
        chase_data = pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-02"],
            "Description": ["Chase Transaction 1", "Chase Transaction 2"],
            "Amount": [100.00, -50.00]
        })
        chase_data.to_csv(chase_csv, index=False)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('cost_allocation.prod.extract.extract_agent_langchain.ChatOpenAI')
    def test_complete_workflow_simulation(self, mock_llm):
        """Test a complete workflow simulation"""
        # Mock LLM responses for different tools
        mock_responses = [
            # Bank decision response
            Mock(content="DECISION: csv_extraction\nREASONING: Chase provides CSV exports\nCONFIDENCE: high"),
            # Data validation response
            Mock(content="QUALITY_SCORE: 9\nCOMPLETENESS: 100%\nISSUES: None\nRECOMMENDATIONS: Good data")
        ]
        mock_llm.return_value.invoke.side_effect = mock_responses
        
        # Create agent with test directory
        agent = LangChainExtractAgent(self.temp_dir)
        
        # Test extraction (this would normally use the full agent, but we're testing the components)
        csv_tool = CSVExtractionTool(self.temp_dir)
        result = csv_tool._run("chase", "*chase*.csv")
        
        self.assertIn("SUCCESS", result)
        self.assertIn("2 transactions", result)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 