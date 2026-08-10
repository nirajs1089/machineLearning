import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_refresh_request import TransactionsRefreshRequest
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
import uuid
import json
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
DEBUG_LOG_PATH = "/Users/vishankagandhi/Documents/git/machineLearning/projects/Financial Planning/cost_allocation/.cursor/debug-f84117.log"
DEBUG_SESSION_ID = "f84117"


def _debug_log(hypothesis_id, location, message, data):
    # #region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": "initial",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


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
                _debug_log(
                    "H4",
                    "unit_test_stp4_v2.py:get_transactions",
                    "transactions_get response metadata",
                    {
                        "start_date": str(start_date.date()),
                        "end_date": str(end_date.date()),
                        "response_total_transactions": response.get("total_transactions"),
                        "batch_count": len(response.get("transactions", [])),
                    },
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
            _debug_log(
                "H4",
                "unit_test_stp4_v2.py:get_transactions",
                "transactions_get raised ApiException",
                {"error": str(e), "start_date": str(start_date.date()), "end_date": str(end_date.date())},
            )
            raise

        return transactions

    def refresh_transactions(self, access_token: str):
        """Trigger Plaid to refresh transaction data for an Item."""
        request = TransactionsRefreshRequest(access_token=access_token)
        return self.rate_limiter.make_request(self.client.transactions_refresh, request)

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
    first_this_month = today.replace(day=1)
    last_prev_month = first_this_month - timedelta(days=1)
    first_prev_month = last_prev_month.replace(day=1)

    first_dt = datetime.combine(first_prev_month, datetime_time.min)
    last_dt = datetime.combine(last_prev_month, datetime_time.min)

    print(first_dt.date())
    print(last_dt.date())
    _debug_log(
        "H5",
        "unit_test_stp4_v2.py:first_and_last_of_previous_month",
        "Computed date window for previous month",
        {"start_date": str(first_dt.date()), "end_date": str(last_dt.date())},
    )

    return first_dt, last_dt


def download_file(private_access_token, bank_name):

    # remove this block once approved fo plaid chase
    # if bank_name == "chase":
    #     import glob

    #     DIRECTORY_PATH = os.getenv("INPUT_FILE_LOCATION")

    #     chase_files = glob.glob(
    #         os.path.join(DIRECTORY_PATH, "*Chase*.csv")
    #     ) + glob.glob(os.path.join(DIRECTORY_PATH, "*Chase*.CSV"))

    #     raw_df = pd.read_csv(chase_files[0])

    #     # 2. Rename selected columns only
    #     # Transaction Date, Post Date, Description, Category, Type, Amount, Memo
    #     # Clean column names (remove extra spaces)
    #     raw_df.columns = raw_df.columns.str.strip()
    #     print(f"Chase columns: {list(raw_df.columns)}")
    #     # transform(raw_df,bank_name)
    #     return raw_df

    plaid_client = PlaidCreditCardClient(
        client_id=PLAID_CLIENT_ID, secret=PLAID_SECRET, environment="production"
    )

    # start_date = datetime.strptime("2025-04-01", '%Y-%m-%d')
    # end_date = datetime.strptime("2025-05-31", '%Y-%m-%d')

    start_date, end_date = first_and_last_of_previous_month()
    _debug_log(
        "H5",
        "unit_test_stp4_v2.py:download_file",
        "Date window and bank used for fetch",
        {
            "bank_name": bank_name,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
        },
    )

    all_transactions = []
    access_tokens = {bank_name: private_access_token}

    # Download from each configured bank
    for bank, access_token in access_tokens.items():
        logger.info(f"Downloading transactions from {bank}")

        try:
            transactions = plaid_client.get_transactions(
                access_token=access_token, start_date=start_date, end_date=end_date
            )

            # New Items: Plaid backfills history asynchronously. If 0 transactions, retry once
            # after a short delay in case sync just finished.
            if len(transactions) == 0:
                logger.warning(
                    f"Got 0 transactions from {bank} for {start_date.date()}–{end_date.date()}. "
                    "If you just re-linked, Plaid may still be syncing. Retrying once in 60s..."
                )
                time.sleep(60)
                transactions = plaid_client.get_transactions(
                    access_token=access_token, start_date=start_date, end_date=end_date
                )
                if len(transactions) == 0:
                    logger.warning(
                        "Still 0 transactions. Wait 15–30 minutes and run again, or check the date range."
                    )

            # Newly linked items can return only recent days at first (e.g., Mar 27-31).
            # If earliest transaction date is after the requested start date, ask Plaid to
            # refresh and retry a few times for historical backfill.
            if transactions:
                earliest_txn_date = min(datetime.strptime(t.date, "%Y-%m-%d").date() for t in transactions)
                requested_start = start_date.date()
                if earliest_txn_date > requested_start:
                    logger.warning(
                        f"Incomplete month for {bank}: earliest={earliest_txn_date}, requested_start={requested_start}. "
                        "Triggering refresh and retrying for historical sync..."
                    )
                    for attempt in range(1, 5):
                        try:
                            plaid_client.refresh_transactions(access_token)
                        except Exception as refresh_error:
                            logger.warning(f"transactions_refresh attempt {attempt} failed: {refresh_error}")

                        time.sleep(20)
                        retry_transactions = plaid_client.get_transactions(
                            access_token=access_token, start_date=start_date, end_date=end_date
                        )
                        if retry_transactions:
                            retry_earliest = min(
                                datetime.strptime(t.date, "%Y-%m-%d").date() for t in retry_transactions
                            )
                            logger.info(
                                f"Retry attempt {attempt}: fetched {len(retry_transactions)} transactions, "
                                f"earliest={retry_earliest}"
                            )
                            transactions = retry_transactions
                            if retry_earliest <= requested_start:
                                break

            # Add bank identifier
            for txn in transactions:
                txn.bank = bank

            all_transactions.extend(transactions)
            logger.info(f"Downloaded {len(transactions)} transactions from {bank}")

        except Exception as e:
            logger.error(f"Error downloading from {bank}: {e}")
            _debug_log(
                "H4",
                "unit_test_stp4_v2.py:download_file",
                "download_file caught exception from get_transactions",
                {"bank_name": bank, "error": str(e), "start_date": str(start_date.date()), "end_date": str(end_date.date())},
            )
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
# first_and_last_of_previous_month()

