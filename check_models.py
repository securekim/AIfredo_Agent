import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
print('API Key loaded:', api_key[:20] + '...' if api_key else 'NOT FOUND')

# Try to list models
try:
    url = 'https://generativelanguage.googleapis.com/v1beta/models?key=' + api_key
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json()
        print('Available models:')
        for model in models.get('models', [])[:10]:  # Show first 10
            print(f'  - {model["name"]}')
    else:
        print(f'Error: {response.status_code} - {response.text}')
except Exception as e:
    print(f'Error: {e}')