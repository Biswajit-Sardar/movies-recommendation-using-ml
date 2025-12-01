import requests
api_key = "da6fad4a" 
movie_name = "Avatar"

url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api_key}"

print(f"Testing URL: {url}")

try:
    response = requests.get(url)
    data = response.json()
    
    print("\n--- Result ---")
    if data.get('Response') == 'True':
        print("Success! Movie Found.")
        print("Poster Link:", data.get('Poster'))
    else:
        print("Error from OMDb:", data.get('Error'))
        print("Possible Reason: API Key might be invalid or typed incorrectly.")
        
except Exception as e:
    print("Connection Failed:", e)
    print("Check your internet connection.")