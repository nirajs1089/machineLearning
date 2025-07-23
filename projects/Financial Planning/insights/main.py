"""
Main Controller for Transaction Chatbot System
Orchestrates the Natural Language to Pandas Operations approach
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any
import argparse
import json

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    GOOGLE_SHEETS_CONFIG, 
    DATA_SCHEMA, 
    LOGGING_CONFIG, 
    CHATBOT_CONFIG,
    ERROR_MESSAGES
)
from data.google_sheets_connector import GoogleSheetsConnector, MockGoogleSheetsConnector
from data.data_processor import DataProcessor
from analysis.query_translator import QueryTranslator, MockQueryTranslator
from analysis.pandas_executor import PandasExecutor
from analysis.result_formatter import ResultFormatter
from chatbot.llm_interface import LLMInterface, MockLLMInterface
from chatbot.conversation_manager import ConversationManager, SessionManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"]),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class TransactionChatbot:
    """
    Main chatbot class that orchestrates all components
    """
    
    def __init__(self, 
                 use_mock_data: bool = False,
                 use_mock_llm: bool = False,
                 spreadsheet_id: Optional[str] = None,
                 credentials_file: Optional[str] = None):
        """
        Initialize the transaction chatbot
        
        Args:
            use_mock_data: Whether to use mock data instead of Google Sheets
            use_mock_llm: Whether to use mock LLM instead of OpenAI
            spreadsheet_id: Google Sheets spreadsheet ID
            credentials_file: Path to Google service account credentials
        """
        self.use_mock_data = use_mock_data
        self.use_mock_llm = use_mock_llm
        self.spreadsheet_id = spreadsheet_id
        self.credentials_file = credentials_file
        
        # Initialize components
        self._initialize_components()
        
        # Initialize session manager
        self.session_manager = SessionManager()
        
        logger.info("Transaction Chatbot initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all system components"""
        try:
            # Step 1: Initialize data connector
            logger.info("Initializing data connector...")
            if self.use_mock_data:
                self.data_connector = MockGoogleSheetsConnector()
                logger.info("Using mock data connector")
            else:
                self.data_connector = GoogleSheetsConnector(
                    spreadsheet_id=self.spreadsheet_id,
                    credentials_file=self.credentials_file
                )
                logger.info("Using Google Sheets connector")
            
            # Step 2: Initialize data processor
            logger.info("Initializing data processor...")
            self.data_processor = DataProcessor()
            
            # Step 3: Initialize LLM interface
            logger.info("Initializing LLM interface...")
            if self.use_mock_llm:
                self.llm_interface = MockLLMInterface()
                logger.info("Using mock LLM interface")
            else:
                self.llm_interface = LLMInterface()
                logger.info("Using OpenAI LLM interface")
            
            # Step 4: Initialize query translator
            logger.info("Initializing query translator...")
            if self.use_mock_llm:
                self.query_translator = MockQueryTranslator()
            else:
                self.query_translator = QueryTranslator(self.llm_interface)
            
            # Step 5: Initialize result formatter
            logger.info("Initializing result formatter...")
            self.result_formatter = ResultFormatter()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def load_transaction_data(self) -> bool:
        """
        Load and process transaction data from Google Sheets
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading transaction data...")
            
            # Step 1: Fetch data from Google Sheets
            logger.info("Step 1: Fetching data from Google Sheets")
            tabs_data = self.data_connector.get_valid_transaction_data(
                DATA_SCHEMA["required_columns"]
            )
            
            if not tabs_data:
                logger.error("No valid transaction data found")
                return False
            
            # Step 2: Process and clean the data
            logger.info("Step 2: Processing and cleaning data")
            self.processed_data = self.data_processor.process_transaction_data(tabs_data)
            
            # Step 3: Initialize pandas executor with processed data
            logger.info("Step 3: Initializing pandas executor")
            self.pandas_executor = PandasExecutor(self.processed_data)
            
            # Step 4: Generate data summary
            logger.info("Step 4: Generating data summary")
            self.data_summary = self.data_processor.get_data_summary(self.processed_data)
            
            logger.info(f"Successfully loaded {len(self.processed_data)} transactions")
            logger.info(f"Data summary: {self.data_summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading transaction data: {e}")
            return False
    
    def create_chat_session(self) -> str:
        """
        Create a new chat session
        
        Returns:
            Session ID
        """
        session_id = self.session_manager.create_session(
            self.query_translator,
            self.pandas_executor,
            self.result_formatter,
            self.llm_interface
        )
        
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Send a message in a specific session
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            Response from the chatbot
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "error": "Session not found",
                "success": False
            }
        
        return session.send_message(message)
    
    def get_welcome_message(self, session_id: str) -> Dict[str, Any]:
        """
        Get welcome message for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Welcome message and examples
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "error": "Session not found",
                "success": False
            }
        
        return session.conversation_manager.get_welcome_message()
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation summary for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation summary
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "error": "Session not found",
                "success": False
            }
        
        return session.conversation_manager.get_conversation_summary()
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a chat session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.session_manager.close_session(session_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status
        
        Returns:
            System status information
        """
        status = {
            "data_loaded": hasattr(self, 'processed_data'),
            "data_summary": getattr(self, 'data_summary', None),
            "llm_available": self.llm_interface.test_connection() if hasattr(self, 'llm_interface') else False,
            "google_sheets_available": self.data_connector.test_connection() if hasattr(self, 'data_connector') else False,
            "session_count": self.session_manager.get_session_count(),
            "active_sessions": self.session_manager.get_active_sessions()
        }
        
        return status
    
    def run_interactive_mode(self) -> None:
        """
        Run the chatbot in interactive mode
        """
        print("=" * 60)
        print("Transaction Chatbot - Interactive Mode")
        print("=" * 60)
        
        # Load data
        if not self.load_transaction_data():
            print("Failed to load transaction data. Exiting.")
            return
        
        # Create session
        session_id = self.create_chat_session()
        welcome = self.get_welcome_message(session_id)
        
        print(f"\n{welcome['response']}")
        if 'examples' in welcome:
            print("\nExample questions:")
            for i, example in enumerate(welcome['examples'], 1):
                print(f"{i}. {example}")
        
        print("\n" + "=" * 60)
        print("Type 'quit' to exit, 'status' for system status, 'summary' for conversation summary")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\nSystem Status: {json.dumps(status, indent=2)}")
                    continue
                elif user_input.lower() == 'summary':
                    summary = self.get_conversation_summary(session_id)
                    print(f"\nConversation Summary: {json.dumps(summary, indent=2)}")
                    continue
                elif not user_input:
                    continue
                
                # Process message
                response = self.send_message(session_id, user_input)
                
                if response.get("success", False):
                    print(f"\nBot: {response['response']}")
                    
                    # Show follow-up questions if available
                    follow_ups = response.get("follow_up_questions", [])
                    if follow_ups:
                        print("\nSuggested follow-up questions:")
                        for i, question in enumerate(follow_ups, 1):
                            print(f"{i}. {question}")
                else:
                    print(f"\nBot: {response.get('response', 'An error occurred.')}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        # Close session
        self.close_session(session_id)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Transaction Chatbot System")
    parser.add_argument("--mock-data", action="store_true", help="Use mock data instead of Google Sheets")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM instead of OpenAI")
    parser.add_argument("--spreadsheet-id", help="Google Sheets spreadsheet ID")
    parser.add_argument("--credentials-file", help="Path to Google service account credentials file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        chatbot = TransactionChatbot(
            use_mock_data=args.mock_data,
            use_mock_llm=args.mock_llm,
            spreadsheet_id=args.spreadsheet_id,
            credentials_file=args.credentials_file
        )

        chatbot.run_interactive_mode()

        # if args.interactive:
        #     # Run interactive mode
        #     chatbot.run_interactive_mode()
        # else:
        #     # Run in API mode (for future web interface)
        #     print("Chatbot initialized. Use --interactive flag to run in interactive mode.")
        #     print("System status:", chatbot.get_system_status())
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 