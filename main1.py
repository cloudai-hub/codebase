import requests
import json

# The URL of the deployed model serving endpoint
url = "https://<databricks-serving-endpoint>/predict"  # replace with your actual URL

# Input data for the model (modify based on your use case)
input_data = {
    "inputs": "Once upon a time in a distant land, there lived a wise old owl."
}

# Headers for the request (if necessary, add authentication tokens here)
headers = {
    "Content-Type": "application/json"
}

# Send the POST request with the input data
response = requests.post(url, headers=headers, data=json.dumps(input_data))

# Check if the request was successful
if response.status_code == 200:
    # Parse the response
    prediction = response.json()
    print("Model Prediction:", prediction)
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")
