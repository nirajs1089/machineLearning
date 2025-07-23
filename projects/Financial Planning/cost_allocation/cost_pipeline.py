import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# directory you want to import from

from extract.unit_test_extract_stp2_3_v1_pri_token import \
    extract
from load.upload_google_sheet import Loader
from transform.openai_allocate_cost import transform

# delete the old chase file for the new month
bank_name="chase"
# bank_name = None  # for plaid flow

raw_df, bank_name = extract(bank_name=bank_name)

transformed_df = transform(raw_df, bank_name)

# add the load step
load = Loader()
load.load_google_sheets(bank_name, transformed_df)
