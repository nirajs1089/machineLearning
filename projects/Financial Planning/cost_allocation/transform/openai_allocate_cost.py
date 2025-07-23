import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import gspread
import numpy as np
import openai
import pandas as pd
from fuzzywuzzy import fuzz
from google.oauth2.service_account import Credentials
from openai import OpenAI
import random  # Add this import for jitter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load.upload_google_sheet import Loader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""WORKS uses open_ai api to allocate costs 
Accuracy is much better """
from enum import Enum

class BankName(Enum):
    BANK_OF_AMERICA = "Bank of America".lower()
    CITIBANK_ONLINE = "Citibank Online".lower()
    CHASE = "chase".lower()

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.interval:
            sleep_time = self.interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class EnhancedTransactionCategorizer:
    def __init__(
        self, directory_path, filter_dates=None, openai_api_key=None, use_gpt4=True
    ):
        """
        Initialize the enhanced transaction categorizer with OpenAI GPT-4 support

        Args:
            directory_path (str): Path to directory containing CSV files
            filter_dates (list): List of date strings in format "M/YYYY" (e.g., ["5/2025", "6/2025"])
            openai_api_key (str): OpenAI API key
            use_gpt4 (bool): Whether to use GPT-4 or fallback to existing methods
        """
        self.directory_path = directory_path
        self.filter_dates = filter_dates or []
        self.training_data = None
        self.chase_data = None
        self.citi_data = None
        self.use_gpt4 = use_gpt4

        # Initialize OpenAI client
        if openai_api_key and use_gpt4:
            self.client = OpenAI(api_key=openai_api_key)
            self.gpt4_available = True

            # At initialization, check available models
            available_models = [model.id for model in self.client.models.list().data]
            preferred_models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
            for m in preferred_models:
                if m in available_models:
                    self.model_name = m
                    break
            else:
                raise Exception("No suitable GPT model available. Please check your OpenAI account.")
        else:
            self.client = None
            self.gpt4_available = False

        # Define standard categories based on your existing patterns
        self.standard_categories = [
            "Home",
            "Groceries",
            "Bills & Utilities",
            "Education",
            "Food & Drink",
            "Travel",
            "Shopping",
            "Health",
            "Travel",
            "gifts",
            "Vacation",
            "Business",
            "Other",
        ]

        # Define common merchant patterns and their categories (keeping existing logic as fallback)
        self.merchant_patterns = {
            "Home": [
                r"amazon\.com",
                r"amazon mktpl",
                r"homeaglow",
                r"costco",
                r"target",
                r"walmart",
                r"home depot",
                r"lowes",
            ],
            "Groceries": [
                r"trader joe",
                r"stop & shop",
                r"whole foods",
                r"cvs",
                r"walgreens",
                r"star osco",
                r"bazaar",
                r"butcherie",
                r"grocery",
                r"market",
            ],
            "Bills & Utilities": [
                r"spotify",
                r"netflix",
                r"autopay",
                r"payment",
                r"electric",
                r"gas company",
                r"verizon",
                r"comcast",
            ],
            "Education": [
                r"openai",
                r"chatgpt",
                r"linkedin",
                r"leetcode",
                r"coursera",
                r"udemy",
                r"books",
                r"thrift books",
            ],
            "Food & Drink": [
                r"tatte",
                r"starbucks",
                r"dunkin",
                r"restaurant",
                r"cafe",
                r"doordash",
                r"grubhub",
                r"uber eats",
            ],
            "Travel": [r"lyft", r"uber", r"greyhound", r"airline", r"hotel", r"airbnb"],
            "Shopping": [r"tjmaxx", r"temu", r"usps", r"ups store", r"fedex"],
        }

        # Cache for GPT-4 responses to avoid repeated API calls for similar transactions
        self.gpt4_cache = {}

        # Confidence thresholds
        self.fuzzy_confidence_threshold = (
            70  # Use GPT-4 if fuzzy matching confidence is below this
        )
        self.pattern_match_threshold = (
            0.8  # Use GPT-4 if pattern matching confidence is below this
        )

        self.rate_limiter = RateLimiter(requests_per_minute=50)  # Conservative limit

    def clean_description(self, description):
        """Clean and normalize description for better matching"""
        if pd.isna(description):
            return ""

        # Convert to string and uppercase
        desc = str(description).upper()

        # Remove common noise patterns
        desc = re.sub(r"NULL XXXXXXXXXXXX\d+", "", desc)
        desc = re.sub(r"AMZN\.COM/BILL WA", "", desc)
        desc = re.sub(
            r"\*[A-Z0-9]+", "", desc
        )  # Remove transaction IDs like *NW6H32630
        desc = re.sub(r"#\d+", "", desc)  # Remove store numbers
        desc = re.sub(r"\d{4}-\d{4}-\d{4}", "", desc)  # Remove phone numbers
        desc = re.sub(r"\s+", " ", desc)  # Normalize whitespace

        return desc.strip()

    def categorize_by_pattern(self, description):
        """Categorize based on merchant patterns with confidence score"""
        clean_desc = self.clean_description(description).lower()

        for category, patterns in self.merchant_patterns.items():
            for pattern in patterns:
                if re.search(pattern, clean_desc):
                    return category, 1.0  # High confidence for pattern matches

        return None, 0.0

    def create_gpt4_prompt(
            self, description: str, examples: List[Dict] = None
    ) -> str:
        """Create an optimized prompt for GPT-4 transaction categorization"""

        categories_str = ", ".join(self.standard_categories)

        prompt = f"""You are an expert financial transaction categorizer. Please categorize the following transaction into one of these categories:

    {categories_str}

    Transaction Description: "{description}"

    Instructions:
    1. Analyze the merchant/description to determine the most appropriate category
    2. Consider common variations in merchant names (e.g., "AMZN MKTP" = Amazon)
    3. If uncertain between categories, choose the most likely one
    4. Respond with ONLY the category name from the list above
    5. If none fit well, use "Other"

    Examples for reference:
    - "STARBUCKS STORE #1234" → Food & Drink
    - "AMAZON.COM AMZN.COM/BILL" → Shopping
    - "WHOLE FOODS MARKET" → Groceries
    - "SHELL OIL" → Gas & Fuel
    - "NETFLIX.COM" → Entertainment
    - "CVS/PHARMACY" → Healthcare (if medicine) or Personal Care (if toiletries)

    {f"Previous similar transactions: {json.dumps(examples, indent=2)}" if examples else ""}

    Category:"""

        return prompt

    def get_similar_examples(
        self, description: str, max_examples: int = 3
    ) -> List[Dict]:
        """Get similar examples from training data to provide context to GPT-4"""
        if self.training_data is None or self.training_data.empty:
            return []

        examples = []
        clean_desc = self.clean_description(description)

        for _, row in self.training_data.iterrows():
            if pd.isna(row["desc"]) or pd.isna(row["type"]):
                continue

            similarity = fuzz.token_set_ratio(
                clean_desc, self.clean_description(row["desc"])
            )
            if similarity > 60:  # Only include reasonably similar examples
                examples.append(
                    {
                        "description": row["desc"],
                        "category": row["type"],
                        "similarity": similarity,
                    }
                )

        # Sort by similarity and return top examples
        examples.sort(key=lambda x: x["similarity"], reverse=True)
        return examples[:max_examples]

    def is_rate_limit_error(self, exception):
        return "429" in str(exception)

    def make_api_call_with_retry(self, prompt: str) -> str:
        """Make API call with retry logic for rate limiting"""
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial transaction categorization expert. Respond with only the category name.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=50,
                    temperature=0.1,
                    timeout=30,
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (random.random() * 0.1)
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"API call failed after {attempt + 1} attempts: {e}")
                    raise e

    def categorize_with_gpt4(
            self, description: str, use_cache: bool = True
    ) -> tuple:
        """Categorize transaction using GPT-4 with caching and error handling"""
        if not self.gpt4_available:
            return None, 0.0

        # Create cache key
        cache_key = f"{self.clean_description(description)}"

        # Check cache first
        if use_cache and cache_key in self.gpt4_cache:
            logger.info(f"Using cached result for: {description[:50]}...")
            return self.gpt4_cache[cache_key], 0.95

        try:
            # Get similar examples for context
            examples = self.get_similar_examples(description)

            # Create prompt
            prompt = self.create_gpt4_prompt(description, examples)

            # Wait if needed to respect rate limits
            self.rate_limiter.wait_if_needed()

            # Replace the direct API call with the rate-limited version
            response_content = self.make_api_call_with_retry(prompt)

            # Validate that the category is in our standard list
            if response_content in self.standard_categories:
                # Cache the result
                if use_cache:
                    self.gpt4_cache[cache_key] = response_content

                logger.info(
                    f"GPT-4 categorized '{description[:50]}...' as '{response_content}'"
                )
                return response_content, 0.95  # High confidence for GPT-4 results
            else:
                logger.warning(
                    f"GPT-4 returned invalid category '{response_content}' for '{description[:30]}...'"
                )
                return "Other", 0.8

        except Exception as e:
            logger.error(f"Error calling GPT-4 for '{description[:30]}...': {str(e)}")
            return None, 0.0

        return None, 0.0


    

    def find_best_category_enhanced(
            self, description, training_descriptions, training_types
    ):
        """Enhanced category finding using GPT-4 when confidence is low"""
        if pd.isna(description):
            return "Other"

        # Step 1: Try pattern-based categorization (fastest)
        pattern_category, pattern_confidence = self.categorize_by_pattern(description)
        if pattern_category and pattern_confidence >= self.pattern_match_threshold:
            logger.info(
                f"Pattern match: '{description[:30]}...' → '{pattern_category}'"
            )
            return pattern_category

        # Step 2: Try fuzzy matching with existing training data
        clean_description = self.clean_description(description)
        best_match_score = 0
        best_category = "Other"

        for train_desc, train_type in zip(training_descriptions, training_types):
            if pd.isna(train_desc) or pd.isna(train_type):
                continue

            clean_train_desc = self.clean_description(train_desc)

            # Use different fuzzy matching techniques
            ratio_score = fuzz.ratio(clean_description, clean_train_desc)
            partial_score = fuzz.partial_ratio(clean_description, clean_train_desc)
            token_sort_score = fuzz.token_sort_ratio(
                clean_description, clean_train_desc
            )
            token_set_score = fuzz.token_set_ratio(clean_description, clean_train_desc)

            # Weight the scores
            combined_score = (
                ratio_score * 0.2
                + partial_score * 0.5
                + token_sort_score * 0.15
                + token_set_score * 0.15
            )

            if combined_score > best_match_score and combined_score > 45:
                best_match_score = combined_score
                best_category = train_type

        # Step 3: Use GPT-4 if confidence is low or no good match found
        if self.gpt4_available and (
            best_match_score < self.fuzzy_confidence_threshold
            or best_category == "Other"
        ):
            logger.info(
                f"Low confidence ({best_match_score:.1f}), using GPT-4 for: '{description[:50]}...'"
            )
            gpt4_category, gpt4_confidence = self.categorize_with_gpt4(
                description
            )

            if gpt4_category and gpt4_confidence > 0.8:
                return gpt4_category

        # Step 4: Return best fuzzy match or pattern match
        return best_category if best_match_score > 45 else (pattern_category or "Other")

    def batch_categorize_with_gpt4(
            self, descriptions: List[str], batch_size: int = 10
    ) -> List[tuple]:
        """Batch process transactions with GPT-4 for better efficiency"""
        if not self.gpt4_available:
            return [(None, 0.0) for _ in descriptions]

        results = []

        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i: i + batch_size]

            logger.info(
                f"Processing GPT-4 batch {i // batch_size + 1}/{(len(descriptions) - 1) // batch_size + 1}"
            )

            batch_results = []
            for desc in batch_descriptions:
                result = self.categorize_with_gpt4(desc)
                batch_results.append(result)

                # Add small delay to respect rate limits
                time.sleep(0.1)

            results.extend(batch_results)

            # Add delay between batches
            if i + batch_size < len(descriptions):
                time.sleep(1)

        return results

    def load_files(self, input_data):
        """Load CSV files from the specified directory"""
        # Find files with keywords in filename

        # citi_files = glob.glob(os.path.join(self.directory_path, "*citi*.csv")) + \
        #              glob.glob(os.path.join(self.directory_path, "*citi*.CSV"))

        training_files = glob.glob(
            os.path.join(self.directory_path, "*cost_training*.csv")
        ) + glob.glob(os.path.join(self.directory_path, "*cost_training*.CSV"))

        # if not chase_files:
        #     raise FileNotFoundError("No Chase CSV files found in directory")
        # if not citi_files:
        #     raise FileNotFoundError("No Citi CSV files found in directory")
        if not training_files:
            raise FileNotFoundError("No cost_training CSV files found in directory")

        # Load the first file found for each type
        # print(f"Loading Chase file: {chase_files[0]}")
        # print(f"Loading Citi file: {citi_files[0]}")
        print(f"Loading Training file: {training_files[0]}")

        self.citi_data = input_data  # pd.read_csv(citi_files[0])
        self.training_data = pd.read_csv(training_files[0])

        self.citi_data.columns = self.citi_data.columns.str.strip()
        self.training_data.columns = self.training_data.columns.str.strip()

        print("Files loaded successfully!")

        print(f"Citi columns: {list(self.citi_data.columns)}")
        print(f"Training columns: {list(self.training_data.columns)}")

    def filter_by_date(self, df, date_column):
        """Filter dataframe by specified dates"""
        if not self.filter_dates:
            return df

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Create filter conditions
        filter_conditions = []
        for date_filter in self.filter_dates:
            try:
                month, year = date_filter.split("/")
                month = int(month)
                year = int(year)
                condition = (df[date_column].dt.month == month) & (
                    df[date_column].dt.year == year
                )
                filter_conditions.append(condition)
            except ValueError:
                print(f"Invalid date format: {date_filter}. Use format 'M/YYYY'")
                continue

        if filter_conditions:
            # Combine all conditions with OR
            combined_condition = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_condition = combined_condition | condition

            return df[combined_condition]

        return df

    def process_citi_data(self, bank_name):
        """Process Citi data with enhanced categorization"""
        # Filter by date if specified

        if bank_name == BankName.CHASE.value or (isinstance(bank_name, str) and bank_name.lower() == "chase"):
            self.citi_data.rename(columns={"Post Date": "Date"}, inplace=True)

        filtered_citi = self.filter_by_date(self.citi_data.copy(), "Date")

        # # Calculate Amount from Debit/Credit
        # filtered_citi['Amount'] = filtered_citi.apply(
        #     lambda row: -float(row['Debit']) if pd.notna(row['Debit']) and row['Debit'] != ''
        #     else float(row['Credit']) if pd.notna(row['Credit']) and row['Credit'] != '' else 0, axis=1
        # )


        bank_patterns = {
            BankName.BANK_OF_AMERICA: [
                "credit",
                "FID BKG",
                "Recurring",
                "AUTOPAY",
                "Fuller Associates",
                "Recurring",
            ],
            BankName.CITIBANK_ONLINE: [
                "AUTOPAY",
            ],
            BankName.CHASE: [
                "AUTOMATIC PAYMENT",
            ],
        }

        # Handle bank_name input - support both Enum and string
        if isinstance(bank_name, BankName):
            # Direct enum input
            patterns = bank_patterns.get(bank_name, [])
        elif isinstance(bank_name, str):
            # String input - try to match enum member name or value
            bank_enum = None
            
            # First try to match by enum member name (e.g., "CITIBANK_ONLINE")
            try:
                bank_enum = BankName[bank_name.upper()]
            except KeyError:
                # If that fails, try to match by enum value (e.g., "citibank online")
                for enum_member in BankName:
                    if enum_member.value == bank_name.lower():
                        bank_enum = enum_member
                        break
            
            if bank_enum:
                patterns = bank_patterns.get(bank_enum, [])
            else:
                # String doesn't match any enum member or value
                patterns = []
                print(f"Warning: Bank name '{bank_name}' not found in BankName enum. Using default patterns.")
        else:
            # Invalid input type
            patterns = []
            print(f"Warning: Invalid bank_name type: {type(bank_name)}. Using default patterns.")

        regex = "|".join(map(re.escape, patterns))
        filtered_citi = filtered_citi[
            ~filtered_citi["Description"].str.contains(regex, case=False, na=False)
        ]
        filtered_citi["Amount"] = filtered_citi["Amount"] * -1

        # Prepare training data
        training_descriptions = self.training_data["desc"].tolist()
        training_types = self.training_data["type"].tolist()

        # Apply enhanced categorization
        print(f"Processing {len(filtered_citi)} Citi transactions...")

        categories = []
        for idx, row in filtered_citi.iterrows():
            category = self.find_best_category_enhanced(
                row["Description"],
                training_descriptions,
                training_types
            )
            categories.append(category)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(filtered_citi)} transactions")

        filtered_citi["type"] = categories

        # Add empty parse column
        filtered_citi["parse"] = ""

        # Select and reorder columns
        result_citi = filtered_citi[
            ["Date", "Description", "Amount", "parse", "type"]
        ].copy()

        # Sort by date in ascending order
        result_citi["Date"] = pd.to_datetime(result_citi["Date"], errors="coerce")
        result_citi = result_citi.sort_values("Date")

        # Convert date back to string for output
        result_citi["Date"] = result_citi["Date"].dt.strftime("%m/%d/%Y")

        return result_citi

    def generate_categorization_report(self, chase_result, citi_result):
        """Generate a report showing categorization results and GPT-4 usage"""
        combined_data = pd.concat(
            [
                chase_result[["Description", "type"]],
                citi_result[["Description", "type"]],
            ],
            ignore_index=True,
        )

        print("\n" + "=" * 60)
        print("CATEGORIZATION REPORT")
        print("=" * 60)

        # Category distribution
        category_counts = combined_data["type"].value_counts()
        print(f"\nTotal transactions processed: {len(combined_data)}")
        print(f"GPT-4 cache size: {len(self.gpt4_cache)}")
        print(f"GPT-4 enabled: {self.gpt4_available}")

        print("\nCategory Distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(combined_data)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # Show some examples of each category
        print("\nSample categorizations:")
        for category in category_counts.head(5).index:
            examples = (
                combined_data[combined_data["type"] == category]["Description"]
                .head(2)
                .tolist()
            )
            print(f"\n{category}:")
            for example in examples:
                print(f"  - {example[:60]}...")


def transform(input_data, bank_name):
    DIRECTORY_PATH = os.getenv("INPUT_FILE_LOCATION")
    FILTER_DATES = []  # ["5/2025", "6/2025"] filtering is done in the extract step

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set this environment variable
    USE_GPT4 = True  # Set to False to use only traditional methods

    if USE_GPT4 and not OPENAI_API_KEY:
        print(
            "Warning: OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )
        print("Falling back to traditional categorization methods.")
        USE_GPT4 = False

    # Initialize the enhanced categorizer
    categorizer = EnhancedTransactionCategorizer(
        DIRECTORY_PATH, FILTER_DATES, openai_api_key=OPENAI_API_KEY, use_gpt4=USE_GPT4
    )

    try:
        # Load files
        categorizer.load_files(input_data)

        print("\nProcessing Citi data with enhanced categorization...")
        citi_result = categorizer.process_citi_data(bank_name)
        print(f"Citi result shape: {citi_result.shape}")

        # Generate report
        # categorizer.generate_categorization_report(chase_result, citi_result)
        # load = Loader()
        # load.load_google_sheets(bank_name, citi_result)

        # Save local copies with timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # chase_result.to_csv(f'chase_processed_gpt4_{timestamp}.csv', index=False)
        # citi_result.to_csv(f'citi_processed_gpt4_{timestamp}.csv', index=False)
        # print(f"\nLocal CSV files saved with timestamp: {timestamp}")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Main execution error: {e}", exc_info=True)

    return citi_result


def compare_results():
    # Run both versions and compare accuracy
    # Original version:
    old_categorizer = TransactionCategorizer(DIRECTORY_PATH, FILTER_DATES)
    old_result = old_categorizer.process_chase_data()

    # Enhanced version:
    new_categorizer = EnhancedTransactionCategorizer(
        DIRECTORY_PATH, FILTER_DATES, use_gpt4=True
    )
    new_result = new_categorizer.process_chase_data()

    # Compare categorizations
    comparison = old_result.merge(
        new_result, on=["Post Date", "Description"], suffixes=("_old", "_new")
    )
    differences = comparison[comparison["type_old"] != comparison["type_new"]]
    print(f"Categorization differences: {len(differences)} out of {len(comparison)}")


def analyze_categorization_performance(result_df):
    """Analyze categorization performance"""
    total = len(result_df)
    categories = result_df["type"].value_counts()

    print(f"Total transactions: {total}")
    print(f"Categories found: {len(categories)}")
    print(
        f"'Other' category: {categories.get('Other', 0)} ({categories.get('Other', 0) / total * 100:.1f}%)"
    )

    # Transactions that might need review
    review_candidates = result_df[result_df["type"] == "Other"]
    print(f"Transactions needing review: {len(review_candidates)}")


class CostTracker:
    def __init__(self):
        self.gpt4_calls = 0
        self.estimated_cost = 0.0

    def track_call(self, input_tokens, output_tokens):
        self.gpt4_calls += 1
        cost = (input_tokens * 0.03 / 1000) + (output_tokens * 0.06 / 1000)
        self.estimated_cost += cost

    def report(self):
        print(f"GPT-4 Calls: {self.gpt4_calls}")
        print(f"Estimated Cost: ${self.estimated_cost:.4f}")


# Periodically review and update training data
def update_training_data(new_transactions_df):
    """Add well-categorized transactions to training data"""
    # Filter high-confidence categorizations
    high_confidence = new_transactions_df[new_transactions_df["confidence"] > 0.9]

    # Add to training data
    training_additions = high_confidence[["Description", "type"]].rename(
        columns={"Description": "desc", "type": "Type"}
    )

    return training_additions


# if __name__ == "__main__":
#     transform()
