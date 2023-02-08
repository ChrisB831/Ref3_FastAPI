'''
Test a POST request to the "https://p3-app.herokuapp.com/inference"
end point....

AUTHOR: Chris Bonham
DATE:   XXXXXXXXXXXXXXXXXXXXX February 2023

To run
1. Make sure app is runnng on Heroku
2. Go to project root
2. python live_post.py
'''
import json
import requests

# Create the correct request body
data = json.dumps({
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
})

# Get the response body by calling the API in the required URL
response = requests.post("https://p3-app.herokuapp.com/inference",
                         data=data)

# Print the response status code
print(response.status_code)

# Print the response body JSON
print(response.json())
