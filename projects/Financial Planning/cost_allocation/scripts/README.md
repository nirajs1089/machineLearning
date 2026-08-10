# Scripts

## Plaid: Remove Item (`plaid_remove_item.py`)

Removes a Plaid Item via `/item/remove`. Use this when you need to **re-link a bank** (e.g. Citibank) so the new Item gets more transaction history (730 days). Once an Item is created, its transaction history length cannot be extended—only by deleting and recreating the Item.

### Quick steps

1. **Get the access token** for the Item to remove (e.g. your Citibank access token from your app or env).

2. **Set env and run** (from project root):

   ```bash
   export PLAID_CLIENT_ID='your_client_id'
   export PLAID_SECRET='your_secret'
   export PLAID_ACCESS_TOKEN_TO_REMOVE='access-production-xxxx...'
   python scripts/plaid_remove_item.py
   ```

   Or pass the token as an argument:

   ```bash
   python scripts/plaid_remove_item.py 'access-production-xxxx...'
   ```

3. **Re-link the bank** using your normal Plaid Link flow (e.g. run the script that opens Link and connect Citibank again). The new Item will get 730 days of transaction history.

4. **Update your app** to use the new `access_token` and `item_id` for that bank.

For full details, see the docstring at the top of `plaid_remove_item.py`.
