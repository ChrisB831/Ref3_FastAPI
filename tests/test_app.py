'''
Test script for the API defined in ./main.py

AUTHOR: Chris Bonham
DATE:   XXXXXXXXXXXXXXXXXXXXX February 2023

This test script is called automatically when the code is pushed to the
remote repo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

To run in isolation
1) Go to the project root
2) python -m pytest ./test/test_app.py -vv
'''
from fastapi.testclient import TestClient
from main import app  # Import the app object from main
import json


# Instantiate a test object
client = TestClient(app)


def test_GET_request():
    '''Tests a GET request to the root end point...
    1) Successfully executes
    2) Returns the expected response body.
       As the response body will change with different models. We check for two
       default keys, the model description and the type of the model used
       NB Checking the response body for the current model is also included for
       completeness

    Arguments: None
    Returns: None
    '''

    # Make a GET request to the endpoint
    r = client.get("/")
    rb = r.json()

    # Test the response
    assert r.status_code == 200
    assert 'Model_description' in rb
    assert "Model_type" in rb
    assert rb == {
        "Model_description": "High salary classifier",
        "Model_type":
            "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
        "n_estimators": 50,
        "split_criterion": "gini",
        "max_depth": 5,
        "n_features": 108
    }


def test_POST_incorrect_rersponse_body():
    '''Test a POST request to the "/inference" end point with an incorrect
    request body

    Arguments: None
    Returns: None
    '''

    # Create the correct request body
    data = {
        "age": "thirty-nine",      # Age incorrectly specified as a string
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
    }

    # Make a POST request to the endpoint and pass the request body
    r = client.post("/inference", data=json.dumps(data))

    # Test the response
    assert r.status_code == 422


def test_POST_request_predict_zero():
    '''Test a POST request to the "/inference" end point....
    1) Successfully executes
    2) The test request body result in a prediction of 0

    Arguments: None
    Returns: None
    '''

    # Create the request body
    data = {
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
    }

    # Make a POST request to the endpoint and pass the request body
    r = client.post("/inference", data=json.dumps(data))

    # Test the response
    assert r.status_code == 200
    assert r.json() == {"prediction": 0}


def test_POST_request_predict_one():
    '''Test a POST request to the "/inference" end point....
    1) Successfully executes
    2) The test request body result in a prediction of 1

    Arguments: None
    Returns: None
    '''

    # Create the request body
    data = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 285066,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }

    # Make a POST request to the endpoint and pass the request body
    r = client.post("/inference", data=json.dumps(data))

    # Test the response
    assert r.status_code == 200
    assert r.json() == {"prediction": 1}
