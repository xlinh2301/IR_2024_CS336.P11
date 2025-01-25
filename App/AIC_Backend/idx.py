import requests

# Define the base URL
base_url = "https://virtserver.swaggerhub.com/NXLINH2301/abc/1.0.0"

# Example: GET request to a specific endpoint
response = requests.get(f"{base_url}/devices")

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed to retrieve data. Status code:", response.status_code)
