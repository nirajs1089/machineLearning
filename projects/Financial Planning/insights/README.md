# Transaction Chatbot System

A sophisticated chatbot system that analyzes credit card transactions using Natural Language to Pandas Operations approach. The system fetches transaction data from Google Sheets, translates natural language queries into pandas operations, executes them locally, and provides insightful responses.

## 🎯 Key Features

- **Natural Language Processing**: Convert user questions into pandas operations
- **Local Data Processing**: Execute operations on dataframes without sending data to LLM
- **Multi-Source Integration**: Fetch data from multiple Google Sheets tabs
- **Cost-Effective**: Minimize LLM API costs by processing data locally
- **Modular Architecture**: Well-structured, testable components
- **Comprehensive Testing**: Unit tests for each component
- **Error Handling**: Robust error handling and fallback mechanisms

## 🏗️ Architecture

### System Components

```
transaction_chatbot/
├── main.py                           # Main controller
├── config/
│   └── settings.py                   # Configuration and constants
├── data/
│   ├── google_sheets_connector.py    # Google Sheets integration
│   └── data_processor.py             # Data processing and cleaning
├── analysis/
│   ├── query_translator.py           # LLM query translation
│   ├── pandas_executor.py            # Pandas operations executor
│   └── result_formatter.py           # Result formatting
├── chatbot/
│   ├── llm_interface.py              # LLM integration
│   └── conversation_manager.py       # Chat session management
├── utils/
│   ├── helpers.py                    # Utility functions
│   └── validators.py                 # Input validation
└── tests/
    ├── test_google_sheets_connector.py
    ├── test_data_processor.py
    ├── test_query_translator.py
    ├── test_pandas_executor.py
    └── test_llm_interface.py
```

### Data Flow

1. **Data Loading**: Fetch transaction data from Google Sheets tabs
2. **Data Processing**: Clean, validate, and prepare data
3. **Query Translation**: Convert natural language to pandas operations
4. **Local Execution**: Execute operations on dataframes
5. **Result Formatting**: Format results for user consumption
6. **Response Generation**: Generate natural language responses

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Sheets API credentials
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd transaction_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export GOOGLE_SPREADSHEET_ID="your_spreadsheet_id"
   export GOOGLE_CREDENTIALS_FILE="path/to/credentials.json"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

4. **Run the chatbot**
   ```bash
   # Interactive mode
   python main.py --interactive
   
   # With mock data (for testing)
   python main.py --interactive --mock-data --mock-llm
   ```

## 📊 Data Schema

The system expects transaction data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| date | Date | Transaction date (YYYY-MM-DD) |
| description | String | Transaction description/vendor |
| amount | Float | Transaction amount |
| category | String | Transaction category |

### Supported Categories

- Food
- Transportation
- Shopping
- Entertainment
- Travel
- Health
- Business
- Gifts

## 💬 Example Queries

The chatbot can handle various types of questions:

### Time-based Analysis
- "Which quarter was the most expensive in the travel category?"
- "Which is the most expensive month in the travel category?"
- "How much did I spend on food last month?"

### Vendor Analysis
- "Which vendor did I spend the most money with?"
- "What are my top 5 spending vendors?"
- "How much did I spend at Amazon this year?"

### Category Analysis
- "What are my top spending categories?"
- "How much did I spend on entertainment vs food?"
- "Which category has the highest average transaction?"

### Comparative Analysis
- "How does my spending this month compare to last month?"
- "What's my spending trend over the last 6 months?"
- "Which quarter had the highest total spending?"

## 🔧 Configuration

### Google Sheets Configuration

```python
GOOGLE_SHEETS_CONFIG = {
    "spreadsheet_name": "Shahdhi Planning",
    "spreadsheet_id": "your_spreadsheet_id",
    "credentials_file": "path/to/credentials.json",
    "tab_names": ["Citi", "Chase", "Cash"],
    "data_range": "A:D"
}
```

### OpenAI Configuration

```python
OPENAI_CONFIG = {
    "api_key": "your_openai_api_key",
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 1000
}
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
python -m pytest tests/test_google_sheets_connector.py -v
python -m pytest tests/test_data_processor.py -v
python -m pytest tests/test_query_translator.py -v
python -m pytest tests/test_pandas_executor.py -v
python -m pytest tests/test_llm_interface.py -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## 🔒 Security Features

### Safe Operations
The system includes security measures to prevent malicious code execution:

- **Operation Validation**: Only allows safe pandas operations
- **Forbidden Operations**: Blocks dangerous operations like `eval()`, `exec()`, `system()`
- **Timeout Protection**: Prevents long-running operations
- **Size Limits**: Prevents memory exhaustion attacks

### Supported Operations
- DataFrame filtering and selection
- GroupBy operations and aggregations
- Date/time operations
- Sorting and ranking
- Basic mathematical operations

## 📈 Performance Optimization

### Cost Efficiency
- **Local Processing**: Data never leaves your system
- **Minimal LLM Calls**: Only translate queries, not process data
- **Caching**: Results can be cached for repeated queries
- **Batch Processing**: Handle multiple queries efficiently

### Scalability
- **Modular Design**: Easy to extend and modify
- **Session Management**: Handle multiple concurrent users
- **Memory Management**: Efficient dataframe operations
- **Error Recovery**: Graceful handling of failures

## 🛠️ Development

### Adding New Query Types

1. **Update Query Translator**
   ```python
   # Add new example queries in config/settings.py
   "new_analysis": {
       "query": "Your new query pattern",
       "operations": [
           "df_filtered = df[df['column'] == 'value']",
           "result = df_filtered.operation()"
       ]
   }
   ```

2. **Update Result Formatter**
   ```python
   # Add new result type handling in analysis/result_formatter.py
   def _format_new_result_type(self, result_value, query):
       # Handle new result type
       pass
   ```

3. **Add Tests**
   ```python
   # Add test cases in tests/test_query_translator.py
   def test_new_query_type(self):
       # Test new query functionality
       pass
   ```

### Extending Data Sources

1. **Create New Connector**
   ```python
   class NewDataSourceConnector:
       def get_transaction_data(self):
           # Implement data fetching
           pass
   ```

2. **Update Main Controller**
   ```python
   # Add new connector to main.py
   if use_new_source:
       self.data_connector = NewDataSourceConnector()
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Google Sheets Connection Error**
   - Verify credentials file path
   - Check spreadsheet ID
   - Ensure proper API permissions

2. **OpenAI API Error**
   - Verify API key
   - Check API quota
   - Ensure internet connection

3. **Data Processing Error**
   - Check data format
   - Verify required columns
   - Review data quality

4. **Query Translation Error**
   - Rephrase the question
   - Use simpler language
   - Check for typos

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 API Reference

### Main Classes

#### TransactionChatbot
Main controller class for the chatbot system.

```python
chatbot = TransactionChatbot(
    use_mock_data=False,
    use_mock_llm=False,
    spreadsheet_id="your_id",
    credentials_file="path/to/credentials.json"
)
```

#### ConversationManager
Manages chat sessions and coordinates components.

```python
session_id = chatbot.create_chat_session()
response = chatbot.send_message(session_id, "Your question here?")
```

#### QueryTranslator
Translates natural language to pandas operations.

```python
translation = translator.translate_query("Which quarter was most expensive?")
```

#### PandasExecutor
Safely executes pandas operations on dataframes.

```python
result = executor.execute_operations([
    "df_filtered = df[df['category'] == 'travel']",
    "result = df_filtered.groupby('quarter')['amount'].sum()"
])
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- Google for the Sheets API
- Pandas community for the excellent data processing library

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Note**: This system is designed for educational and personal use. Always ensure compliance with data privacy regulations when handling financial data. 