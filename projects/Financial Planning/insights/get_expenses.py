#!/usr/bin/env python3
"""
Standalone script for getting monthly average expenses by category
Can be called from external projects via subprocess
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data.expense_aggregator import ExpenseAggregator
    
    def main():
        """Main function to get expenses and output JSON"""
        try:
            # Parse command line arguments
            if len(sys.argv) < 3:
                print(json.dumps({
                    'success': False,
                    'error': 'Usage: python get_expenses.py <start_date> <end_date> [years]'
                }))
                return
            
            # Get date parameters
            start_date_str = sys.argv[1]
            end_date_str = sys.argv[2]
            years = int(sys.argv[3]) if len(sys.argv) > 3 else None
            
            # Parse dates
            start_date = datetime.fromisoformat(start_date_str)
            end_date = datetime.fromisoformat(end_date_str)
            
            # Initialize aggregator
            aggregator = ExpenseAggregator()
            
            # Get monthly average expenses
            if years:
                expenses_df = aggregator.get_monthly_average_expenses(years=years)
            else:
                expenses_df = aggregator.get_monthly_average_expenses(
                    start_date=start_date,
                    end_date=end_date
                )
            
            # Convert to JSON-serializable format
            if not expenses_df.empty:
                result = {
                    'success': True,
                    'data': expenses_df.to_dict('records'),
                    'total_expenses': float(expenses_df['Actual'].sum()),
                    'category_count': len(expenses_df)
                }
            else:
                result = {
                    'success': False,
                    'data': [],
                    'error': 'No data found'
                }
            
            print(json.dumps(result))
            
        except Exception as e:
            result = {
                'success': False,
                'data': [],
                'error': str(e)
            }
            print(json.dumps(result))
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(json.dumps({
        'success': False,
        'data': [],
        'error': f'Failed to import ExpenseAggregator: {str(e)}'
    })) 