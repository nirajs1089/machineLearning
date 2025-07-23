import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.country_code import CountryCode
from plaid.model.products import Products
from plaid.configuration import Configuration
from plaid.api_client import ApiClient
from datetime import datetime, timedelta
import logging
import time, os
from plaid.exceptions import ApiException
from datetime import date, datetime, timedelta, time as datetime_time
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Optional, Any
from ratelimit import limits, sleep_and_retry
import sys



# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transform.openai_allocate_cost import transform

"""Create a link token for Plaid Link initialization"""
user_id = "test_user_123"

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = plaid.Environment.Production  # Use sandbox environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransactionData:
    """Data class for transaction information"""

    account_id: str
    account_name: str
    transaction_id: str
    date: str
    description: str
    amount: float
    category: List[str]
    merchant_name: Optional[str] = None
    bank: str = ""


class RateLimitHandler:
    """Handles rate limiting for API calls"""

    def __init__(self, calls_per_minute: int = 100):
        self.calls_per_minute = calls_per_minute
        self.last_reset = time.time()
        self.calls_made = 0

    @sleep_and_retry
    @limits(calls=100, period=60)  # 100 calls per minute
    def make_request(self, func, *args, **kwargs):
        """Rate-limited API request wrapper"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Rate limit hit, sleeping: {e}")
            time.sleep(1)
            raise


class PlaidCreditCardClient:
    """Enhanced Plaid client for credit card transactions"""

    def __init__(self, client_id: str, secret: str, environment: str = "sandbox"):
        self.client_id = client_id
        self.secret = secret

        # Set environment
        if environment == "sandbox":
            host = "https://sandbox.plaid.com"
        elif environment == "development":
            host = "https://development.plaid.com"
        elif environment == "production":
            host = "https://production.plaid.com"
        else:
            raise ValueError(
                "Environment must be 'sandbox', 'development', or 'production'"
            )

        configuration = Configuration(
            host=host, api_key={"clientId": client_id, "secret": secret}
        )

        # Create API client with retry strategy
        api_client = ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)
        self.rate_limiter = RateLimitHandler()

    def get_transactions(
        self,
        access_token: str,
        start_date: datetime,
        end_date: datetime,
        account_ids=None,
    ):
        """Get transactions for specified date range"""

        transactions = []
        offset = 0
        batch_size = 500  # Plaid's max per request

        try:
            while True:
                request = TransactionsGetRequest(
                    access_token=access_token,
                    start_date=start_date.date(),
                    end_date=end_date.date(),
                    # offset=offset,
                    # count=batch_size,
                    # account_ids=account_ids
                )

                response = self.rate_limiter.make_request(
                    self.client.transactions_get, request
                )

                batch_transactions = response["transactions"]

                # Convert to TransactionData objects
                for txn in batch_transactions:
                    transaction_data = TransactionData(
                        account_id=txn["account_id"],
                        account_name=txn.get("account_owner", "Unknown"),
                        transaction_id=txn["transaction_id"],
                        date=txn["date"].strftime("%Y-%m-%d"),
                        description=txn["name"],
                        amount=-txn["amount"],  # Plaid uses negative for debits
                        category=txn.get("category", []),
                        merchant_name=txn.get("merchant_name"),
                        bank=self._identify_bank(txn.get("account_id", "")),
                    )
                    transactions.append(transaction_data)

                # Check if we've got all transactions
                if len(batch_transactions) < batch_size:
                    break

                offset += batch_size

                # Respect rate limits
                time.sleep(0.1)

        except ApiException as e:
            logger.error(f"Error fetching transactions: {e}")
            raise

        return transactions

    def _identify_bank(self, account_id: str) -> str:
        """Identify bank from account ID pattern"""
        # This is a simplified approach - in reality, you'd map account IDs to banks
        if "chase" in account_id.lower():
            return "Chase"
        elif "citi" in account_id.lower():
            return "Citi"
        else:
            return "Unknown"


def first_and_last_of_previous_month() -> tuple[str, str]:
    """
    Returns (first_day, last_day) of the previous month
    in YYYY-MM-DD format.
    """
    today = date.today()
    first_this_month = today.replace(day=1)  # e.g. 2025-06-01
    last_prev_month = first_this_month - timedelta(days=1)  # e.g. 2025-05-31
    first_prev_month = last_prev_month.replace(day=1)  # e.g. 2025-05-01

    # Convert the `date` objects to `datetime` at midnight
    first_dt = datetime.combine(first_prev_month, datetime_time.min)
    last_dt = datetime.combine(last_prev_month, datetime_time.min)

    return first_dt, last_dt


def download_file(private_access_token, bank_name):

    # remove this block once approved fo plaid chase
    if bank_name == "chase":
        import glob

        DIRECTORY_PATH = os.getenv("INPUT_FILE_LOCATION")

        chase_files = glob.glob(
            os.path.join(DIRECTORY_PATH, "*Chase*.csv")
        ) + glob.glob(os.path.join(DIRECTORY_PATH, "*Chase*.CSV"))

        raw_df = pd.read_csv(chase_files[0])

        # 2. Rename selected columns only
        # Transaction Date, Post Date, Description, Category, Type, Amount, Memo
        # Clean column names (remove extra spaces)
        raw_df.columns = raw_df.columns.str.strip()
        print(f"Chase columns: {list(raw_df.columns)}")
        # transform(raw_df,bank_name)
        return raw_df

    plaid_client = PlaidCreditCardClient(
        client_id=PLAID_CLIENT_ID, secret=PLAID_SECRET, environment="production"
    )

    # start_date = datetime.strptime("2025-04-01", '%Y-%m-%d')
    # end_date = datetime.strptime("2025-05-31", '%Y-%m-%d')

    start_date, end_date = first_and_last_of_previous_month()

    all_transactions = []
    access_tokens = {bank_name: private_access_token}

    # Download from each configured bank
    for bank, access_token in access_tokens.items():
        logger.info(f"Downloading transactions from {bank}")

        try:
            transactions = plaid_client.get_transactions(
                access_token=access_token, start_date=start_date, end_date=end_date
            )

            # Add bank identifier
            for txn in transactions:
                txn.bank = bank

            all_transactions.extend(transactions)
            logger.info(f"Downloaded {len(transactions)} transactions from {bank}")

        except Exception as e:
            logger.error(f"Error downloading from {bank}: {e}")
            continue

    records = []  # list that will hold each row-dict

    for txn in all_transactions:  # iterate normally

        # if txn.merchant_name is None:
        #     description = txn.description
        # else:
        #     description = txn.merchant_name

        description = txn.description

        row = {
            # 'bank': txn.bank,
            # 'account_id': txn.account_id,
            "Date": txn.date,
            # 'description': txn.description,
            "Amount": txn.amount,
            # 'category': txn.category, #', '.join(txn.category),
            "Description": description,
        }
        records.append(row)  # add it to the list

    # create the DataFrame after the loop
    raw_df = pd.DataFrame(records)

    # Save to CSV
    # filename = f"{bank_name.lower()}_transactions_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    # raw_df.to_csv(filename, index=False)

    logger.info(f"Successfully downloaded {len(all_transactions)} transactions")

    # transform(raw_df,bank_name)
    return raw_df


# create_link_token_api()

