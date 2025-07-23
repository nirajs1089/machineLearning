import sys

# directory you want to import from
extra_dir = "/Users/vishankagandhi/Documents/"


# 2️⃣  OR insert at the front (highest priority)
sys.path.insert(0, str(extra_dir))
from cost_allocation.prod.extract.unit_test_stp4_v2 import download_file
from cost_allocation.prod.load.upload_google_sheet import Loader
from cost_allocation.prod.transform.openai_allocate_cost import transform

from openai import OpenAI

# bank_name="chase"
bank_name = "Citibank Online"  # for plaid flow

map = {
    "Citibank Online": "access-production-22f765ec-73b9-4944-97b4-546366a52bbf",
    "Bank of America": "access-production-d6d2732f-7910-43f0-809c-e5020822993a",
}

raw_df = download_file(map[bank_name], bank_name)

transformed_df = transform(raw_df, bank_name)

# Save to CSV
filename = f"{bank_name.lower()}_processed_transactions.csv"
transformed_df.to_csv(filename, index=False)

# # add the load step
# load = Loader()
# load.load_google_sheets(bank_name, transformed_df)
