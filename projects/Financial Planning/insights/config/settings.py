"""
Configuration settings for the Transaction Chatbot System
Using Natural Language to Pandas Operations approach
"""

import os
from typing import List, Dict, Any

# Google Sheets Configuration
GOOGLE_SHEETS_CONFIG = {
    "spreadsheet_name": "Shahdhi Planning",
    "spreadsheet_id": "1_WdlG7nwnHOsdCwhbM0AsM7psoU9g_oBXEEyezK3UmQ",# or os.getenv("GOOGLE_SPREADSHEET_ID", ""),
    "credentials_file": "/Users/vishankagandhi/Documents/cost_allocation/nirajs1089-11250649e9d3.json",# or os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"),
    "scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
    "tab_names": ["Citi", "Chase", "Cash"],  # Configurable tab names
    "data_range": "B:F"  # Configurable range for data
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 1000
}

# Data Schema Configuration
DATA_SCHEMA = {
    "required_columns": ["date", "description", "amount", "category"],
    "date_format": "%m/%d/%Y",
    "amount_column": "amount",
    "category_column": "category",
    "description_column": "description",
    "date_column": "date"
}

# Query Translation Configuration
QUERY_TRANSLATION_CONFIG = {
    "system_prompt": """You are a query translation expert. Convert natural language queries about financial transactions into pandas operations.

Available columns in the dataframe:
- date: Transaction date
- description: Transaction description
- amount: Transaction amount
- category: Transaction category
- tab_name: Source tab name (Citi, Chase, Cash)

Available pandas operations:
- Filtering: df[df['column'] == 'value'], df[df['amount'] > 100]
- Grouping: df.groupby('column')
- Aggregation: .sum(), .mean(), .count(), .max(), .min()
- Sorting: .sort_values('column', ascending=False)
- Date operations: pd.to_datetime(), .dt.month, .dt.quarter, .dt.year

When filtering a DataFrame, always use .copy() after filtering to avoid SettingWithCopyWarning.

Do not use any import statements. Assume all necessary libraries (pandas, numpy) are already imported.

Return only the JSON response with pandas_operations array.""",
    
    "example_queries": {
        "quarterly_analysis": {
            "query": "Which quarter was the most expensive in the travel category?",
            "operations": [
                "df_filtered = df[df['category'].str.strip().str.lower() == 'travel']",
                "df_filtered['quarter'] = pd.to_datetime(df_filtered['date']).dt.quarter",
                "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
                "result = df_grouped.idxmax()"
            ]
        },
        "monthly_analysis": {
            "query": "Which is the most expensive month in the travel category?",
            "operations": [
                "df_filtered = df[df['category'].str.strip().str.lower() == 'travel']",
                "df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month",
                "df_grouped = df_filtered.groupby('month')['amount'].sum()",
                "result = df_grouped.idxmax()"
            ]
        },
        "vendor_analysis": {
            "query": "Which vendor did I spend the most money with?",
            "operations": [
                "df_grouped = df.groupby('description')['amount'].sum()",
                "result = df_grouped.idxmax()"
            ]
        },
        "top_n_expenses_may_travel": {
            "query": "filter on month of may and travel, give me the top 5 expenses, include the year and month for each of the top expenses",
            "operations": [
                "df_filtered = df[df['category'].str.strip().str.lower() == 'travel']",
                "df_filtered['year'] = pd.to_datetime(df_filtered['date']).dt.year",
                "df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month",
                "df_filtered_may = df_filtered[df_filtered['month'] == 5]",
                "result = df_filtered_may.sort_values('amount', ascending=False).head(5)"
            ]
        },
        "last_month_travel": {
            "query": "How much did I spend on travel last month?",
            "operations": [
                "import datetime",
                "today = pd.Timestamp.today()",
                "first_day_this_month = today.replace(day=1)",
                "last_month_end = first_day_this_month - pd.Timedelta(days=1)",
                "last_month_start = last_month_end.replace(day=1)",
                "df_filtered = df[df['category'].str.strip().str.lower() == 'travel']",
                "df_filtered = df_filtered[(df_filtered['date'] >= last_month_start) & (df_filtered['date'] <= last_month_end)]",
                "result = df_filtered['amount'].sum()"
            ]
        }
    }
}

# Pandas Operations Configuration
PANDAS_OPERATIONS_CONFIG = {
    "safe_operations": [
        "filter", "groupby", "sum", "mean", "count", "max", "min", "std",
        "sort_values", "head", "tail", "reset_index", "rename",
        "to_datetime", "dt.month", "dt.quarter", "dt.year", "dt.day"
    ],
    "forbidden_operations": [
        "eval", "exec", "system", "subprocess", "os.", "import",
        "delete", "drop", "modify", "write", "save"
    ],
    "max_operations": 10,  # Maximum number of operations per query
    "timeout_seconds": 300  # Maximum execution time
}

# Chatbot Configuration
CHATBOT_CONFIG = {
    "max_conversation_history": 10,
    "system_prompt": """You are a helpful financial analysis assistant. You can analyze credit card transactions and provide insights about spending patterns, trends, and comparisons. Always provide clear, actionable insights with specific numbers and percentages when possible.""",
    "welcome_message": "Hello! I'm your financial analysis assistant. I can help you analyze your credit card transactions. Ask me questions like 'Which quarter was the most expensive in the travel category?' or 'How much did I spend on food last month?'"
}

# Error Messages
ERROR_MESSAGES = {
    "missing_credentials": "Google Sheets credentials not found. Please check your configuration.",
    "invalid_spreadsheet": "Unable to access the specified Google Spreadsheet.",
    "missing_columns": "Required columns not found in the spreadsheet.",
    "invalid_date_format": "Date format in spreadsheet is not recognized.",
    "no_data_found": "No transaction data found for the specified criteria.",
    "invalid_query": "I couldn't understand your question. Please try rephrasing it.",
    "api_error": "An error occurred while processing your request.",
    "unsafe_operation": "The requested operation is not allowed for security reasons.",
    "execution_timeout": "The query took too long to execute. Please try a simpler question.",
    "translation_error": "Unable to translate your question into operations. Please try rephrasing."
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "transaction_chatbot.log"
}

# Test Configuration
TEST_CONFIG = {
    "test_spreadsheet_id": os.getenv("TEST_SPREADSHEET_ID", ""),
    "test_data_file": "test_data.csv",
    "mock_responses": {
        "quarterly_travel": {
            "pandas_operations": [
                "df_filtered = df[df['category'] == 'Travel']",
                "df_filtered['quarter'] = pd.to_datetime(df_filtered['date']).dt.quarter",
                "df_grouped = df_filtered.groupby('quarter')['amount'].sum()",
                "result = df_grouped.idxmax()"
            ]
        }
    }
} 