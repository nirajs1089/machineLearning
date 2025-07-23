"""
LangChain Agentic AI Extract Workflow
Intelligently extracts credit card transactions from different banks using LangChain agents and tools.
"""

import os
import sys
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import glob
from datetime import datetime, date, timedelta
import json
import re

# LangChain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Add the parent directory to path for imports
sys.path.insert(0, "/Users/vishankagandhi/Documents/")

# Import existing extraction modules
from cost_allocation.prod.extract.unit_test_extract_stp2_3_v1_pri_token import PlaidLocalLink
from cost_allocation.prod.extract.unit_test_stp4_v2 import PlaidCreditCardClient, first_and_last_of_previous_month

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankType(Enum):
    """Enum for different bank types and their extraction methods"""
    PLAID_API = "plaid_api"
    CSV_FILE = "csv_file"


class ExtractionMethod(Enum):
    """Available extraction methods"""
    PLAID_LINK_FLOW = "plaid_link_flow"
    PLAID_DIRECT_API = "plaid_direct_api"
    CSV_READER = "csv_reader"


@dataclass
class BankConfig:
    """Configuration for each bank"""
    name: str
    type: BankType
    extraction_method: ExtractionMethod
    csv_pattern: Optional[str] = None
    access_token: Optional[str] = None
    plaid_institution_id: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of extraction operation"""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    bank_name: Optional[str] = None
    error_message: Optional[str] = None
    extraction_method: Optional[str] = None
    transaction_count: int = 0


class CSVExtractionTool(BaseTool):
    """Tool for extracting transactions from CSV files"""
    
    name = "csv_extraction"
    description = "Extract credit card transactions from CSV files. Use this for banks that provide CSV exports like Chase."
    
    def __init__(self, base_directory: str = "/Users/vishankagandhi/Documents/cost_allocation"):
        super().__init__()
        self.base_directory = base_directory
    
    def _run(self, bank_name: str, csv_pattern: str = None) -> str:
        """Extract transactions from CSV files"""
        try:
            if not csv_pattern:
                csv_pattern = f"*{bank_name}*.csv"
            
            # Find CSV files matching the pattern
            full_pattern = os.path.join(self.base_directory, csv_pattern)
            csv_files = glob.glob(full_pattern) + glob.glob(full_pattern.replace(".csv", ".CSV"))
            
            if not csv_files:
                return f"ERROR: No CSV files found matching pattern: {full_pattern}"
            
            # Read the first matching CSV file
            csv_file = csv_files[0]
            raw_df = pd.read_csv(csv_file)
            
            # Clean column names
            raw_df.columns = raw_df.columns.str.strip()
            
            # Save to temporary file for the agent to access
            temp_file = f"/tmp/{bank_name.lower()}_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            raw_df.to_csv(temp_file, index=False)
            
            return f"SUCCESS: Extracted {len(raw_df)} transactions from {csv_file}. Data saved to {temp_file}"
            
        except Exception as e:
            return f"ERROR: CSV extraction failed: {str(e)}"
    
    async def _arun(self, bank_name: str, csv_pattern: str = None) -> str:
        """Async version of CSV extraction"""
        return self._run(bank_name, csv_pattern)


class PlaidExtractionTool(BaseTool):
    """Tool for extracting transactions using Plaid API"""
    
    name = "plaid_extraction"
    description = "Extract credit card transactions using Plaid API. Use this for banks that support Plaid integration like Citibank and Bank of America."
    
    def __init__(self):
        super().__init__()
        self.plaid_client_id = os.getenv("PLAID_CLIENT_ID")
        self.plaid_secret = os.getenv("PLAID_SECRET")
        
        if not self.plaid_client_id or not self.plaid_secret:
            logger.warning("Plaid credentials not found in environment variables")
    
    def _run(self, bank_name: str, access_token: str = None) -> str:
        """Extract transactions using Plaid API"""
        try:
            if not self.plaid_client_id or not self.plaid_secret:
                return "ERROR: Plaid credentials not configured"
            
            if access_token:
                # Use direct API with access token
                from cost_allocation.prod.extract.unit_test_stp4_v2 import download_file
                raw_df = download_file(access_token, bank_name.lower())
                
                if raw_df is not None:
                    temp_file = f"/tmp/{bank_name.lower()}_plaid_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    raw_df.to_csv(temp_file, index=False)
                    return f"SUCCESS: Extracted {len(raw_df)} transactions using Plaid API. Data saved to {temp_file}"
                else:
                    return "ERROR: No data returned from Plaid API"
            else:
                # Use interactive Plaid Link flow
                plaid_link = PlaidLocalLink(
                    client_id=self.plaid_client_id,
                    secret=self.plaid_secret,
                    environment="production"
                )
                
                result = plaid_link.run_link_flow(
                    user_id=f"user_{bank_name.lower().replace(' ', '_')}",
                    client_name=f"Cost Allocation - {bank_name}",
                    products=["transactions"],
                    port=8080
                )
                
                if not result:
                    return "ERROR: Plaid Link flow failed or was cancelled"
                
                # Get metadata and download transactions
                metadata = json.loads(result["metadata"])
                connected_bank_name = metadata["institution"]["name"]
                
                from cost_allocation.prod.extract.unit_test_stp4_v2 import download_file
                raw_df = download_file(result["access_token"], bank_name.lower())
                
                if raw_df is not None:
                    temp_file = f"/tmp/{connected_bank_name.lower().replace(' ', '_')}_plaid_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    raw_df.to_csv(temp_file, index=False)
                    return f"SUCCESS: Extracted {len(raw_df)} transactions from {connected_bank_name} using Plaid Link. Data saved to {temp_file}"
                else:
                    return "ERROR: No data returned from Plaid API"
                    
        except Exception as e:
            return f"ERROR: Plaid extraction failed: {str(e)}"
    
    async def _arun(self, bank_name: str, access_token: str = None) -> str:
        """Async version of Plaid extraction"""
        return self._run(bank_name, access_token)


class IntelligentBankDecisionTool(BaseTool):
    """AI-powered tool for intelligently deciding extraction methods"""
    
    name = "intelligent_bank_decision"
    description = "Use AI reasoning to decide the best extraction method for any bank, considering multiple factors like bank capabilities, data availability, and user preferences."
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.decision_history = []
        self.learning_data = []
    
    def _run(self, bank_name: str, context: str = "") -> str:
        """Use AI to intelligently decide extraction method"""
        try:
            # Create a comprehensive prompt for AI reasoning
            prompt = f"""
You are an expert financial data extraction specialist. Analyze the following bank and context to determine the optimal extraction method.

Bank Name: {bank_name}
Context: {context}

Available extraction methods:
1. csv_extraction - For banks that provide CSV exports (manual file upload)
2. plaid_extraction - For banks that support Plaid API integration (automated)

Consider these factors in your decision:
- Bank's technical capabilities and API support
- Data freshness requirements
- Security and compliance considerations
- User experience preferences
- Historical success rates
- Cost implications
- Data quality and completeness

Previous successful extractions:
- Chase: csv_extraction (provides reliable CSV exports)
- Citibank: plaid_extraction (excellent Plaid integration)
- Bank of America: plaid_extraction (good Plaid support)
- Wells Fargo: plaid_extraction (recent Plaid integration)
- American Express: csv_extraction (limited API access)

For unknown banks, consider:
- Bank size and technology adoption
- Common industry practices
- User's preference for automation vs manual processes

Provide your reasoning and recommendation in this format:
DECISION: [method]
REASONING: [detailed explanation]
CONFIDENCE: [high/medium/low]
ALTERNATIVES: [other methods to consider]
"""
            
            # Get AI decision
            response = self.llm.invoke(prompt)
            decision = response.content
            
            # Store decision for learning
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "bank_name": bank_name,
                "context": context,
                "decision": decision
            })
            
            return decision
            
        except Exception as e:
            return f"ERROR: AI decision failed: {str(e)}"
    
    async def _arun(self, bank_name: str, context: str = "") -> str:
        """Async version of AI decision"""
        return self._run(bank_name, context)
    
    def learn_from_feedback(self, bank_name: str, method_used: str, success: bool, feedback: str = ""):
        """Learn from extraction results to improve future decisions"""
        self.learning_data.append({
            "bank_name": bank_name,
            "method_used": method_used,
            "success": success,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update decision patterns based on feedback
        if success:
            logger.info(f"Learning: {method_used} worked well for {bank_name}")
        else:
            logger.warning(f"Learning: {method_used} failed for {bank_name}, will consider alternatives")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data"""
        if not self.learning_data:
            return {"message": "No learning data available"}
        
        total_attempts = len(self.learning_data)
        successful_attempts = sum(1 for d in self.learning_data if d["success"])
        success_rate = successful_attempts / total_attempts
        
        # Analyze method effectiveness
        method_stats = {}
        for data in self.learning_data:
            method = data["method_used"]
            if method not in method_stats:
                method_stats[method] = {"total": 0, "success": 0}
            method_stats[method]["total"] += 1
            if data["success"]:
                method_stats[method]["success"] += 1
        
        return {
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "method_effectiveness": method_stats,
            "recent_decisions": self.decision_history[-5:]
        }


class AdaptiveExtractionTool(BaseTool):
    """Tool that adapts extraction strategy based on real-time feedback"""
    
    name = "adaptive_extraction"
    description = "Intelligently adapt extraction strategy based on real-time results and learning from previous attempts."
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.extraction_attempts = []
        self.success_patterns = {}
    
    def _run(self, bank_name: str, initial_method: str, failure_reason: str = "") -> str:
        """Adapt extraction strategy based on results"""
        try:
            # Record the attempt
            self.extraction_attempts.append({
                "bank_name": bank_name,
                "method": initial_method,
                "success": not bool(failure_reason),
                "failure_reason": failure_reason,
                "timestamp": datetime.now().isoformat()
            })
            
            if not failure_reason:
                # Success - update success patterns
                if bank_name not in self.success_patterns:
                    self.success_patterns[bank_name] = []
                self.success_patterns[bank_name].append(initial_method)
                return f"SUCCESS: {initial_method} worked for {bank_name}"
            
            # Failure - use AI to suggest alternative strategy
            prompt = f"""
Previous extraction attempt failed:
Bank: {bank_name}
Method tried: {initial_method}
Failure reason: {failure_reason}

Available methods:
- csv_extraction: Manual CSV file upload
- plaid_extraction: Automated Plaid API integration

Based on the failure and bank characteristics, suggest the best alternative approach:

1. Should we try a different method?
2. Are there specific parameters we should adjust?
3. Should we try a hybrid approach?

Consider:
- Bank's technical limitations
- Common failure patterns
- Alternative data sources
- User preferences for manual vs automated processes

Provide your recommendation in this format:
RECOMMENDATION: [method]
REASONING: [explanation]
PARAMETERS: [any specific parameters to adjust]
"""
            
            response = self.llm.invoke(prompt)
            recommendation = response.content
            
            return f"ADAPTATION: {recommendation}"
            
        except Exception as e:
            return f"ERROR: Adaptive extraction failed: {str(e)}"
    
    async def _arun(self, bank_name: str, initial_method: str, failure_reason: str = "") -> str:
        """Async version of adaptive extraction"""
        return self._run(bank_name, initial_method, failure_reason)
    
    def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights about adaptation patterns"""
        if not self.extraction_attempts:
            return {"message": "No adaptation data available"}
        
        # Analyze failure patterns
        failures = [a for a in self.extraction_attempts if not a["success"]]
        success_rate = 1 - (len(failures) / len(self.extraction_attempts))
        
        # Common failure reasons
        failure_reasons = {}
        for failure in failures:
            reason = failure["failure_reason"]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            "total_attempts": len(self.extraction_attempts),
            "success_rate": success_rate,
            "failure_patterns": failure_reasons,
            "success_patterns": self.success_patterns
        }


class DataValidationTool(BaseTool):
    """Tool for validating extracted data"""
    
    name = "data_validation"
    description = "Validate extracted transaction data for completeness and quality"
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _run(self, file_path: str) -> str:
        """Validate extracted data using AI"""
        try:
            if not os.path.exists(file_path):
                return f"ERROR: File not found: {file_path}"
            
            df = pd.read_csv(file_path)
            
            # Use AI to analyze data quality
            prompt = f"""
Analyze this transaction dataset for quality and completeness:

Dataset Info:
- Rows: {len(df)}
- Columns: {list(df.columns)}
- Sample data: {df.head(3).to_dict()}

Check for:
1. Data completeness (missing values)
2. Data consistency (format issues)
3. Data relevance (appropriate transaction data)
4. Data freshness (recent transactions)
5. Potential anomalies or errors

Provide a comprehensive quality assessment in this format:
QUALITY_SCORE: [1-10]
COMPLETENESS: [percentage]
ISSUES: [list of issues found]
RECOMMENDATIONS: [suggestions for improvement]
"""
            
            response = self.llm.invoke(prompt)
            ai_analysis = response.content
            
            # Basic validation checks
            validation_results = []
            
            # Check for required columns
            required_columns = ["Date", "Description", "Amount"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results.append(f"Missing required columns: {missing_columns}")
            
            # Check for empty dataframe
            if len(df) == 0:
                validation_results.append("DataFrame is empty")
            
            # Check for null values in critical columns
            for col in required_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        validation_results.append(f"Column '{col}' has {null_count} null values")
            
            if validation_results:
                return f"VALIDATION ISSUES: {'; '.join(validation_results)}\n\nAI ANALYSIS:\n{ai_analysis}"
            else:
                return f"VALIDATION PASSED: {len(df)} transactions extracted successfully\n\nAI ANALYSIS:\n{ai_analysis}"
                
        except Exception as e:
            return f"ERROR: Validation failed: {str(e)}"
    
    async def _arun(self, file_path: str) -> str:
        """Async version of data validation"""
        return self._run(file_path)


class ExtractAgentPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the extract agent"""
    
    template = """You are an intelligent financial data extraction agent with advanced reasoning capabilities. Your job is to extract credit card transactions from different banks using the most appropriate method.

Available tools:
{tools}

You have access to:
1. Intelligent bank decision making
2. Multiple extraction methods (CSV, Plaid API)
3. Adaptive strategies based on real-time feedback
4. AI-powered data validation
5. Learning from previous attempts

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do, considering multiple factors
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""

    def format(self, **kwargs) -> str:
        """Format the prompt"""
        tools = kwargs.get("tools", [])
        tool_names = [tool.name for tool in tools]
        return self.template.format(
            tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
            tool_names=", ".join(tool_names),
            **kwargs
        )


class ExtractAgentOutputParser(BaseOutputParser):
    """Parser for the extract agent output"""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the agent output"""
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Parse action and input
        action_match = re.search(r"Action: (\w+)", text)
        input_match = re.search(r"Action Input: (.+)", text)
        
        if action_match and input_match:
            action = action_match.group(1)
            action_input = input_match.group(1).strip()
            return AgentAction(tool=action, tool_input=action_input, log=text)
        
        raise ValueError(f"Could not parse LLM output: {text}")


class LangChainExtractAgent:
    """
    LangChain-based agentic AI workflow for extracting credit card transactions.
    """
    
    def __init__(self, base_directory: str = "/Users/vishankagandhi/Documents/cost_allocation"):
        self.base_directory = base_directory
        self.extraction_history = []
        
        # Initialize tools with intelligent capabilities
        self.tools = [
            IntelligentBankDecisionTool(),
            CSVExtractionTool(base_directory),
            PlaidExtractionTool(),
            AdaptiveExtractionTool(),
            DataValidationTool()
        ]
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent"""
        prompt = ExtractAgentPromptTemplate(
            tools=self.tools,
            input_variables=["input", "agent_scratchpad"]
        )
        
        output_parser = ExtractAgentOutputParser()
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        tool_names = [tool.name for tool in self.tools]
        
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=ConversationBufferMemory(memory_key="chat_history")
        )
    
    def extract(self, bank_name: Optional[str] = None, export_csv: bool = False) -> Tuple[pd.DataFrame, str]:
        """
        Main extraction method using intelligent LangChain agent.
        
        Args:
            bank_name: Name of the bank to extract from
            export_csv: Whether to export the result to CSV file
            
        Returns:
            Tuple of (DataFrame, bank_name) - compatible with existing pipeline
        """
        if not bank_name:
            bank_name = "unknown"
        
        # Record extraction attempt
        self.extraction_history.append({
            "timestamp": datetime.now().isoformat(),
            "bank_name": bank_name,
            "status": "started"
        })
        
        try:
            # Use intelligent LangChain agent to handle extraction
            result = self.agent.run(
                f"Intelligently extract credit card transactions from {bank_name}. "
                f"Use AI reasoning to decide the best method, adapt if needed, and validate the results. "
                f"{'Export the result to CSV file.' if export_csv else ''}"
            )
            
            # Parse the result to get the file path
            if "Data saved to" in result:
                file_path = result.split("Data saved to")[-1].strip()
                df = pd.read_csv(file_path)
                
                # Extract bank name from file path or use provided name
                extracted_bank_name = bank_name
                if "plaid_extracted" in file_path:
                    # Try to extract bank name from file path
                    filename = os.path.basename(file_path)
                    if "_" in filename:
                        extracted_bank_name = filename.split("_")[0]
                
                # Update extraction history
                self.extraction_history[-1].update({
                    "status": "success",
                    "transaction_count": len(df),
                    "file_path": file_path
                })
                
                logger.info(f"✅ Successfully extracted {len(df)} transactions from {extracted_bank_name}")
                
                return df, extracted_bank_name
            else:
                raise Exception(f"Extraction failed: {result}")
                
        except Exception as e:
            # Update extraction history
            self.extraction_history[-1].update({
                "status": "failed",
                "error": str(e)
            })
            
            logger.error(f"❌ Extraction failed: {e}")
            raise
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction history"""
        if not self.extraction_history:
            return {"message": "No extractions performed yet"}
        
        total_extractions = len(self.extraction_history)
        successful_extractions = sum(1 for h in self.extraction_history if h["status"] == "success")
        total_transactions = sum(h.get("transaction_count", 0) for h in self.extraction_history if h["status"] == "success")
        
        return {
            "total_extractions": total_extractions,
            "successful_extractions": successful_extractions,
            "success_rate": successful_extractions / total_extractions if total_extractions > 0 else 0,
            "total_transactions": total_transactions,
            "recent_extractions": self.extraction_history[-5:]
        }
    
    def get_intelligence_insights(self) -> Dict[str, Any]:
        """Get insights from intelligent tools"""
        insights = {}
        
        # Get insights from intelligent bank decision tool
        decision_tool = next((tool for tool in self.tools if isinstance(tool, IntelligentBankDecisionTool)), None)
        if decision_tool:
            insights["bank_decisions"] = decision_tool.get_learning_insights()
        
        # Get insights from adaptive extraction tool
        adaptive_tool = next((tool for tool in self.tools if isinstance(tool, AdaptiveExtractionTool)), None)
        if adaptive_tool:
            insights["adaptation_patterns"] = adaptive_tool.get_adaptation_insights()
        
        return insights


# Convenience function for backward compatibility
def extract(bank_name=None, export_csv=False):
    """
    Backward-compatible extract function that uses the intelligent LangChain agentic AI workflow.
    
    Args:
        bank_name: Name of the bank to extract from
        export_csv: Whether to export the result to CSV file
        
    Returns:
        Tuple of (DataFrame, bank_name) - compatible with existing pipeline
    """
    agent = LangChainExtractAgent()
    return agent.extract(bank_name, export_csv)


if __name__ == "__main__":
    # Example usage
    agent = LangChainExtractAgent()
    
    # Extract from Chase (CSV)
    print("🔄 Extracting from Chase using intelligent LangChain agent...")
    df_chase, bank_chase = agent.extract("chase", export_csv=True)
    print(f"✅ Extracted {len(df_chase)} transactions from {bank_chase}")
    
    # Show extraction summary
    summary = agent.get_extraction_summary()
    print(f"📊 Extraction Summary: {summary}")
    
    # Show intelligence insights
    insights = agent.get_intelligence_insights()
    print(f"🧠 Intelligence Insights: {insights}") 