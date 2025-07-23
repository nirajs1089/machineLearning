import os
import webbrowser
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.configuration import Configuration
from plaid.model.products import Products
from plaid.api_client import ApiClient


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract.unit_test_stp4_v2 import download_file

"""gets the access token for Citi"""


class PlaidLinkHandler(BaseHTTPRequestHandler):
    """HTTP request handler for capturing Plaid Link responses"""

    def do_GET(self):
        """Handle GET requests from Plaid Link"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/":
            # Serve the Plaid Link HTML page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            html_content = self.server.get_link_html()
            self.wfile.write(html_content.encode())

        elif parsed_path.path == "/success":
            # Handle successful link
            query_params = parse_qs(parsed_path.query)
            public_token = query_params.get("public_token", [None])[0]
            metadata = query_params.get("metadata", [None])[0]

            if public_token:
                self.server.public_token = public_token
                self.server.metadata = metadata

                # Send success response
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                success_html = """
                <html>
                <head><title>Plaid Link Success</title></head>
                <body>
                    <h2>✅ Bank Account Connected Successfully!</h2>
                    <p>You can now close this browser window.</p>
                    <script>
                        setTimeout(() => window.close(), 3000);
                    </script>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode())

                # Signal completion
                self.server.link_completed = True

        elif parsed_path.path == "/error":
            # Handle link errors
            query_params = parse_qs(parsed_path.query)
            error = query_params.get("error", [None])[0]

            self.server.error = error
            self.server.link_completed = True

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            error_html = f"""
            <html>
            <head><title>Plaid Link Error</title></head>
            <body>
                <h2>❌ Connection Failed</h2>
                <p>Error: {error or 'Unknown error occurred'}</p>
                <p>You can close this browser window.</p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())

    def log_message(self, format, *args):
        """Suppress server logs"""
        pass


class PlaidLinkServer(HTTPServer):
    """Custom HTTP server for Plaid Link integration"""

    def __init__(self, server_address, handler_class, link_token):
        super().__init__(server_address, handler_class)
        self.link_token = link_token
        self.public_token = None
        self.metadata = None
        self.error = None
        self.link_completed = False

    def get_link_html(self):
        """Generate HTML page with Plaid Link integration"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connect Your Bank Account</title>
            <script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 400px;
                }}
                .btn {{
                    background: #00d4aa;
                    color: white;
                    border: none;
                    padding: 16px 32px;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: background 0.3s;
                }}
                .btn:hover {{
                    background: #00b894;
                }}
                h1 {{
                    color: #2d3748;
                    margin-bottom: 20px;
                }}
                p {{
                    color: #4a5568;
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏦 Connect Your Bank</h1>
                <p>Click the button below to securely connect your bank account using Plaid.</p>
                <button id="link-button" class="btn">Connect Bank Account</button>
            </div>

            <script>
                const linkHandler = Plaid.create({{
                    token: '{self.link_token}',
                    onSuccess: (public_token, metadata) => {{
                        console.log('Link success:', public_token, metadata);
                        // Redirect to success endpoint with token
                        window.location.href = `/success?public_token=${{public_token}}&metadata=${{encodeURIComponent(JSON.stringify(metadata))}}`;
                    }},
                    onExit: (err, metadata) => {{
                        console.log('Link exit:', err, metadata);
                        if (err) {{
                            window.location.href = `/error?error=${{encodeURIComponent(err.error_message || 'User exited')}}`;
                        }} else {{
                            window.close();
                        }}
                    }},
                    onEvent: (eventName, metadata) => {{
                        console.log('Link event:', eventName, metadata);
                    }}
                }});

                document.getElementById('link-button').onclick = () => {{
                    linkHandler.open();
                }};

                // Auto-open Link after page loads
                setTimeout(() => {{
                    linkHandler.open();
                }}, 1000);
            </script>
        </body>
        </html>
        """


class PlaidLocalLink:
    """Main class for handling local Plaid Link integration"""

    def __init__(self, client_id, secret, environment="production"):
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

        # Configure Plaid client
        configuration = Configuration(
            host=host, api_key={"clientId": client_id, "secret": secret}
        )
        api_client = ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)

    def create_link_token(
        self, user_id, client_name="Local Python App", products=["transactions"]
    ):
        """Create a link token for Plaid Link"""
        try:
            user = LinkTokenCreateRequestUser(client_user_id=user_id)

            request = LinkTokenCreateRequest(
                products=[Products("transactions")],
                client_name=client_name,
                country_codes=[CountryCode("US")],
                language="en",
                user=user,
                # redirect_uri='http://localhost:8080/success'  # Local redirect
            )

            response = self.client.link_token_create(request)
            return response["link_token"]

        except Exception as e:
            raise Exception(f"Failed to create link token: {str(e)}")

    def exchange_public_token(self, public_token):
        """Exchange public token for access token"""
        try:
            request = ItemPublicTokenExchangeRequest(public_token=public_token)
            response = self.client.item_public_token_exchange(request)

            return {
                "access_token": response["access_token"],
                "item_id": response["item_id"],
            }

        except Exception as e:
            raise Exception(f"Failed to exchange public token: {str(e)}")

    def run_link_flow(
        self,
        user_id,
        client_name="Local Python App",
        products=["transactions"],
        port=8080,
    ):
        """
        Run the complete Plaid Link flow locally

        Returns:
            dict: Contains access_token, item_id, and metadata
        """
        print("🚀 Starting Plaid Link flow...")

        # Step 1: Create link token
        print("📝 Creating link token...")
        try:
            link_token = self.create_link_token(user_id, client_name, products)
            print("✅ Link token created successfully")
        except Exception as e:
            print(f"❌ Failed to create link token: {e}")
            return None
        # link_token = "link-production-9108ea54-28ad-4b2f-a2cf-5012dc4a99d0"
        # Step 2: Start local server
        print(f"🌐 Starting local server on port {port}...")
        server = PlaidLinkServer(("localhost", port), PlaidLinkHandler, link_token)

        # Run server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # Step 3: Open browser
        url = f"http://localhost:{port}"
        print(f"🌍 Opening browser at {url}")
        webbrowser.open(url)

        # Step 4: Wait for completion
        print("⏳ Waiting for user to complete bank connection...")
        print("   (The browser will open automatically)")

        # Poll for completion
        timeout = 300  # 5 minutes timeout
        start_time = time.time()

        while not server.link_completed and (time.time() - start_time) < timeout:
            time.sleep(1)

        # Stop server
        server.shutdown()

        if server.error:
            print(f"❌ Link failed: {server.error}")
            return None

        if not server.public_token:
            print("❌ Link timed out or was cancelled")
            return None

        # Step 5: Exchange public token
        print("🔄 Exchanging public token for access token...")
        try:
            result = self.exchange_public_token(server.public_token)
            print("✅ Successfully obtained access token!")

            return {
                "access_token": result["access_token"],
                "item_id": result["item_id"],
                "public_token": server.public_token,
                "metadata": server.metadata,
            }

        except Exception as e:
            print(f"❌ Failed to exchange public token: {e}")
            return None


# Example usage
def extract(bank_name=None):
    """Example of how to use the PlaidLocalLink class"""
    if bank_name == "chase":
        return download_file(None, bank_name), bank_name

    # Set your Plaid credentials
    PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
    PLAID_SECRET = os.getenv("PLAID_SECRET")

    # if CLIENT_ID == 'your_client_id_here' or SECRET == 'your_secret_here':
    #     print("❌ Please set your PLAID_CLIENT_ID and PLAID_SECRET environment variables")
    #     print("   export PLAID_CLIENT_ID='your_actual_client_id'")
    #     print("   export PLAID_SECRET='your_actual_secret'")
    #     return

    # Create PlaidLocalLink instance
    plaid_link = PlaidLocalLink(
        client_id=PLAID_CLIENT_ID,
        secret=PLAID_SECRET,
        environment="production",  # Change to 'production' for live
    )

    # Run the link flow
    result = plaid_link.run_link_flow(
        user_id="local_user_123",
        client_name="My Python App",
        products=["transactions", "auth", "identity"],
        port=8080,  # You can change the port if needed
    )

    if result:
        metadata = json.loads(result["metadata"])

        bank_name = metadata["institution"]["name"]
        print("\n🎉 SUCCESS! Bank account connected!")
        print(f"private Access Token: {result['access_token']}...")
        print(f"Logged in to Bank {bank_name}")
        print(f"Item ID: {result['item_id']}")
        print("\nYou can now use the access_token to make API calls to Plaid")

        # Example: You could now fetch account info
        # accounts = plaid_link.client.accounts_get(...)
        time.sleep(3)
        return download_file(result["access_token"], bank_name.lower()), bank_name.lower()

    else:
        print("\n❌ Failed to connect bank account")


# if __name__ == "__main__":
#     extract()
