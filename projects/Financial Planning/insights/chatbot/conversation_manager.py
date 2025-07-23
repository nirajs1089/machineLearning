"""
Conversation Manager for handling chat sessions
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CHATBOT_CONFIG
from analysis.query_translator import QueryTranslator
from analysis.pandas_executor import PandasExecutor
from analysis.result_formatter import ResultFormatter
from chatbot.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation sessions and coordinates between components
    """
    
    def __init__(self, 
                 query_translator: QueryTranslator,
                 pandas_executor: PandasExecutor,
                 result_formatter: ResultFormatter,
                 llm_interface: LLMInterface):
        """
        Initialize the conversation manager
        
        Args:
            query_translator: QueryTranslator instance
            pandas_executor: PandasExecutor instance
            result_formatter: ResultFormatter instance
            llm_interface: LLMInterface instance
        """
        self.query_translator = query_translator
        self.pandas_executor = pandas_executor
        self.result_formatter = result_formatter
        self.llm_interface = llm_interface
        self.conversation_history = []
        self.max_history = CHATBOT_CONFIG["max_conversation_history"]
        self.session_start_time = datetime.now()
        
        logger.info("Conversation manager initialized")
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response
        
        Args:
            user_message: User's input message
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Add user message to history
            self._add_to_history("user", user_message)
            
            # Step 1: Translate natural language to pandas operations
            logger.info("Step 1: Translating query to pandas operations")
            translation_result = self.query_translator.translate_query(user_message)
            
            if translation_result.get("error"):
                error_response = f"I couldn't understand your question: {translation_result['error']}. Please try rephrasing it."
                self._add_to_history("assistant", error_response)
                return {
                    "response": error_response,
                    "translation_result": translation_result,
                    "execution_result": None,
                    "formatted_result": None,
                    "follow_up_questions": [],
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 2: Execute pandas operations
            logger.info("Step 2: Executing pandas operations")
            pandas_operations = translation_result.get("pandas_operations", [])
            execution_result = self.pandas_executor.execute_operations(pandas_operations)
            
            if not execution_result.get("success", False):
                error_response = f"I encountered an error while analyzing your data: {execution_result.get('error', 'Unknown error')}. Please try a different question."
                self._add_to_history("assistant", error_response)
                return {
                    "response": error_response,
                    "translation_result": translation_result,
                    "execution_result": execution_result,
                    "formatted_result": None,
                    "follow_up_questions": [],
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 3: Format results
            logger.info("Step 3: Formatting results")
            formatted_result = self.result_formatter.format_execution_result(
                execution_result, 
                user_message, 
                translation_result
            )
            
            # Step 4: Generate natural language response
            logger.info("Step 4: Generating natural language response")
            llm_response = self.llm_interface.generate_response(
                query=user_message,
                formatted_result=formatted_result,
                conversation_history=self.conversation_history
            )
            
            # Step 5: Generate follow-up questions
            logger.info("Step 5: Generating follow-up questions")
            follow_ups = self.llm_interface.generate_follow_up_questions(
                query=user_message,
                formatted_result=formatted_result
            )
            
            # Add bot response to history
            self._add_to_history("assistant", llm_response)
            
            # Prepare response
            result = {
                "response": llm_response,
                "translation_result": translation_result,
                "execution_result": execution_result,
                "formatted_result": formatted_result,
                "follow_up_questions": follow_ups,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Processed message successfully: {len(llm_response)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # Add error response to history
            error_response = "I'm sorry, I encountered an error while processing your request. Please try again."
            self._add_to_history("assistant", error_response)
            
            return {
                "response": error_response,
                "translation_result": None,
                "execution_result": None,
                "formatted_result": None,
                "follow_up_questions": [],
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_welcome_message(self) -> Dict[str, Any]:
        """
        Get the welcome message for new conversations
        
        Returns:
            Dictionary with welcome message and initial suggestions
        """
        welcome_msg = CHATBOT_CONFIG["welcome_message"]
        
        # Get supported query examples
        supported_queries = self.query_translator.get_example_queries()
        examples = list(supported_queries.values())[:3]  # Show first 3 examples
        
        return {
            "response": welcome_msg,
            "examples": examples,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation
        
        Returns:
            Dictionary with conversation summary
        """
        if not self.conversation_history:
            return {"message": "No conversation history available"}
        
        # Count messages by role
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        
        # Get session duration
        session_duration = datetime.now() - self.session_start_time
        
        # Get unique topics (based on query types)
        topics = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                # Extract potential topics from user messages
                content = msg["content"].lower()
                if "quarter" in content:
                    topics.append("Quarterly Analysis")
                elif "month" in content:
                    topics.append("Monthly Analysis")
                elif "vendor" in content:
                    topics.append("Vendor Analysis")
                elif "category" in content:
                    topics.append("Category Analysis")
                elif "travel" in content:
                    topics.append("Travel Spending")
        
        unique_topics = list(set(topics))
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "session_duration_minutes": session_duration.total_seconds() / 60,
            "topics_discussed": unique_topics,
            "session_start": self.session_start_time.isoformat()
        }
    
    def clear_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        self.session_start_time = datetime.now()
        logger.info("Conversation history cleared")
    
    def _add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history
        
        Args:
            role: Role of the message sender ('user' or 'assistant')
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(message)
        
        # Maintain history size limit
        if len(self.conversation_history) > self.max_history * 2:  # *2 because each exchange has 2 messages
            # Keep the most recent messages
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """
        Get recent messages from the conversation
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Export the conversation for analysis or storage
        
        Returns:
            Dictionary with conversation data
        """
        return {
            "session_start": self.session_start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_messages": len(self.conversation_history),
            "messages": self.conversation_history,
            "summary": self.get_conversation_summary()
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of pandas execution history
        
        Returns:
            Execution summary
        """
        return self.pandas_executor.get_execution_summary()


class ChatSession:
    """
    Represents a single chat session
    """
    
    def __init__(self, session_id: str, conversation_manager: ConversationManager):
        """
        Initialize a chat session
        
        Args:
            session_id: Unique identifier for the session
            conversation_manager: ConversationManager instance
        """
        self.session_id = session_id
        self.conversation_manager = conversation_manager
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
    
    def send_message(self, message: str) -> Dict[str, Any]:
        """
        Send a message in this session
        
        Args:
            message: User message
            
        Returns:
            Response from the conversation manager
        """
        self.last_activity = datetime.now()
        return self.conversation_manager.process_message(message)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get session status information
        
        Returns:
            Dictionary with session status
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "duration_minutes": (datetime.now() - self.created_at).total_seconds() / 60
        }
    
    def close(self) -> None:
        """Close the session"""
        self.is_active = False
        self.last_activity = datetime.now()


class SessionManager:
    """
    Manages multiple chat sessions
    """
    
    def __init__(self):
        """Initialize the session manager"""
        self.sessions = {}
        self.session_counter = 0
    
    def create_session(self, 
                      query_translator: QueryTranslator,
                      pandas_executor: PandasExecutor,
                      result_formatter: ResultFormatter,
                      llm_interface: LLMInterface) -> str:
        """
        Create a new chat session
        
        Args:
            query_translator: QueryTranslator instance
            pandas_executor: PandasExecutor instance
            result_formatter: ResultFormatter instance
            llm_interface: LLMInterface instance
            
        Returns:
            Session ID
        """
        self.session_counter += 1
        session_id = f"session_{self.session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conversation_manager = ConversationManager(
            query_translator, 
            pandas_executor, 
            result_formatter, 
            llm_interface
        )
        session = ChatSession(session_id, conversation_manager)
        
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            ChatSession instance or None if not found
        """
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was closed, False if not found
        """
        session = self.sessions.get(session_id)
        if session:
            session.close()
            logger.info(f"Closed session: {session_id}")
            return True
        return False
    
    def cleanup_inactive_sessions(self, max_inactive_minutes: int = 60) -> int:
        """
        Clean up inactive sessions
        
        Args:
            max_inactive_minutes: Maximum minutes of inactivity before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if not session.is_active:
                continue
            
            inactive_minutes = (current_time - session.last_activity).total_seconds() / 60
            if inactive_minutes > max_inactive_minutes:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.close_session(session_id)
            del self.sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
        
        return len(sessions_to_remove)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sessions
        
        Returns:
            List of session information dictionaries
        """
        active_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_active:
                active_sessions.append(session.get_status())
        
        return active_sessions
    
    def get_session_count(self) -> Dict[str, int]:
        """
        Get session count statistics
        
        Returns:
            Dictionary with session counts
        """
        total_sessions = len(self.sessions)
        active_sessions = len([s for s in self.sessions.values() if s.is_active])
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "inactive_sessions": total_sessions - active_sessions
        } 