#!/usr/bin/env python3
"""
Remove a Plaid Item via /item/remove.

Use this when you need to re-link a bank (e.g. Citibank) so the new Item gets
more transaction history (transactions.days_requested set at link time). Once
removed, run the Plaid Link flow again to create a new Item with 730 days.

INSTRUCTIONS
------------
1. Get the access_token for the Item you want to remove:
   - From your app: it's the token you use for that bank when calling
     get_transactions (e.g. the Citibank access token).
   - Or from Plaid Dashboard: Items → select Item → (access token may be
     shown or you use the one you stored when you linked).

2. Run this script with that access_token:

   Option A – environment variable (recommended; avoids token in shell history):
     export PLAID_ACCESS_TOKEN_TO_REMOVE='access-production-xxxx...'
     python scripts/plaid_remove_item.py

   Option B – command-line argument:
     python scripts/plaid_remove_item.py 'access-production-xxxx...'

3. Ensure PLAID_CLIENT_ID and PLAID_SECRET are set (same as your pipeline):
   export PLAID_CLIENT_ID='...'
   export PLAID_SECRET='...'

4. After removal, re-link the bank:
   - Run your normal Plaid Link flow (e.g. the script that uses PlaidLocalLink
     and run_link_flow, or cost_pipeline with interactive link).
   - Connect the same bank again. The new Item will get 730 days of
     transaction history (if you use the updated link token with
     transactions=LinkTokenTransactions(days_requested=730)).

5. Update your app to use the new access_token and item_id for that bank.
"""

import argparse
import os
import sys

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plaid.api import plaid_api
from plaid.model.item_remove_request import ItemRemoveRequest
from plaid.configuration import Configuration
from plaid.api_client import ApiClient


def get_plaid_client():
    """Build Plaid API client using env credentials."""
    client_id = os.getenv("PLAID_CLIENT_ID")
    secret = os.getenv("PLAID_SECRET")
    if not client_id or not secret:
        print("Error: Set PLAID_CLIENT_ID and PLAID_SECRET in the environment.")
        sys.exit(1)
    host = "https://production.plaid.com"
    configuration = Configuration(
        host=host, api_key={"clientId": client_id, "secret": secret}
    )
    api_client = ApiClient(configuration)
    return plaid_api.PlaidApi(api_client)


def remove_item(access_token: str) -> None:
    """Call Plaid /item/remove for the given access_token."""
    client = get_plaid_client()
    request = ItemRemoveRequest(access_token=access_token)
    response = client.item_remove(request)
    # ItemRemoveResponse has request_id; success means item was removed
    print("Item removed successfully.")
    if getattr(response, "request_id", None):
        print(f"Request ID: {response.request_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove a Plaid Item (e.g. to re-link with more transaction history).",
        epilog="See script docstring for full instructions.",
    )
    parser.add_argument(
        "access_token",
        nargs="?",
        default=os.getenv("PLAID_ACCESS_TOKEN_TO_REMOVE"),
        help="Access token of the Item to remove (or set PLAID_ACCESS_TOKEN_TO_REMOVE)",
    )
    args = parser.parse_args()

    if not args.access_token or not args.access_token.strip():
        print("Error: Provide access_token via PLAID_ACCESS_TOKEN_TO_REMOVE or as argument.")
        parser.print_help()
        sys.exit(1)

    access_token = args.access_token.strip()
    remove_item(access_token)


if __name__ == "__main__":
    main()
