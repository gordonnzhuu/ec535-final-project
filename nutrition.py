"""
test_nutritionix_credentials.py

A simple script to verify your Nutritionix API credentials by making a test request.

Usage:
  1. Install dependencies:
       pip install requests
  2. Set environment variables for your credentials (recommended):
       export NIX_APP_ID="your_app_id"
       export NIX_APP_KEY="your_app_key"
     Or replace the placeholders in the script below.
  3. Run:
       python test_nutritionix_credentials.py
"""

import sys
import requests

# Retrieve credentials from environment or use placeholders
APP_ID = "fdee5b24"
APP_KEY = "72a92983f8200a50bf667a30653c0e24"

# Basic validation
if APP_ID == 'YOUR_APP_ID' or APP_KEY == 'YOUR_APP_KEY':
    print("Error: Please set your Nutritionix APP_ID and APP_KEY in environment variables "
          "(NIX_APP_ID, NIX_APP_KEY) or update the placeholders in this script.")
    sys.exit(1)

# Test endpoint (natural nutrients)
url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
data = {'query': 'apple and banana and mapo tofu'}  # sample query
headers = {
    'x-app-id': APP_ID,
    'x-app-key': APP_KEY,
    'Content-Type': 'application/json'
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=5)
    print(f"HTTP Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Success! Nutrition facts for 'apple':")
        nutrition_data = response.json()
        for food in nutrition_data.get('foods', []):
            print(f"Food: {food['food_name']}")
            print(f"Serving Size: {food['serving_qty']} {food['serving_unit']}")
            print(f"Calories: {food['nf_calories']} kcal")
            print(f"Total Fat: {food['nf_total_fat']} g")
            print(f"Protein: {food['nf_protein']} g")
            print(f"Carbohydrates: {food['nf_total_carbohydrate']} g")
            print(f"Sugars: {food['nf_sugars']} g")
            print(f"Fiber: {food['nf_dietary_fiber']} g")
            print("-" * 30)
    else:
        print("❌ Failure. Response body:")
        print(response.text)
except requests.RequestException as e:
    print(f"❌ Request failed: {e}")