
import gspread
from google.oauth2.service_account import Credentials
import time
import os

class Loader:

    def __init__(self):
        self.gc = None

    def load_google_sheets(self,bank_name,result):

        # Configuration
        SPREADSHEET_NAME = "Shahdhi Planning"
        # CREDENTIALS_FILE = f"{DIRECTORY_PATH}/nirajs1089-11250649e9d3.json"
        CREDENTIALS_FILE = os.getenv("GOOGLE_API_KEY_LOCATION")

        # Setup Google Sheets and upload
        print("\nSetting up Google Sheets...")
        if self.setup_google_sheets_credentials(CREDENTIALS_FILE):
            time.sleep(3)
            self.upload_to_google_sheets(SPREADSHEET_NAME, bank_name,result) #chase_result
        else:
            print("Skipping Google Sheets upload due to credential issues")

    def setup_google_sheets_credentials(self, credentials_file_path):
        """Setup Google Sheets API credentials"""
        try:
            # Define the scope
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']

            # Load credentials
            credentials = Credentials.from_service_account_file(credentials_file_path, scopes=scope)

            # Authorize the client
            self.gc = gspread.authorize(credentials)
            print("Google Sheets credentials setup successfully!")
            return True

        except Exception as e:
            print(f"Error setting up credentials: {e}")
            print("Please ensure you have:")
            print("1. Created a service account in Google Cloud Console")
            print("2. Downloaded the JSON credentials file")
            print("3. Shared your Google Sheet with the service account email")
            return False


    def upload_to_google_sheets(self, spreadsheet_name, bank_name, citi_result):  # chase_result
        """Upload results to Google Sheets"""
        try:
            # Open the spreadsheet
            spreadsheet = self.gc.open(spreadsheet_name)

            # # Process Chase data - get existing worksheet
            # chase_sheet = spreadsheet.worksheet('ChaseTest')
            #
            # # Find first empty row in column B
            # chase_values = chase_sheet.col_values(2)  # Column B
            # chase_start_row = len([v for v in chase_values if v.strip()]) + 1
            #
            # # Upload Chase data
            # chase_data_to_upload = chase_result.values.tolist()
            # if chase_data_to_upload:
            #     chase_range = f'B{chase_start_row}:E{chase_start_row + len(chase_data_to_upload) - 1}'
            #     chase_sheet.update(chase_range, chase_data_to_upload)
            #     print(f"Chase data uploaded to {chase_range}")

            bank_tab_map = {"Citibank Online": "Citi",
                            "chase": "Chase",
                            "Bank of America": "Cash"}

            # Process Citi data - get existing worksheet
            citi_sheet = spreadsheet.worksheet(bank_tab_map[bank_name])

            # Find first empty row in column B
            citi_values = citi_sheet.col_values(2)  # Column B
            citi_start_row = len([v for v in citi_values if v.strip()]) + 1

            # Upload Citi data
            citi_data_to_upload = citi_result.values.tolist()
            if citi_data_to_upload:
                citi_range = f'B{citi_start_row}:F{citi_start_row + len(citi_data_to_upload) - 1}'
                citi_sheet.update(citi_range, citi_data_to_upload)
                print(f"{bank_name} data uploaded to {citi_range}")

            print(f"{bank_name} Data successfully uploaded to Google Sheets! under tab {bank_tab_map[bank_name]}")

        except Exception as e:
            print(f"Error uploading to Google Sheets: {e}")