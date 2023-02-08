'''
TBD

Very basic test script for the app containing 3 test
NB Pytest must be installed

To run, go to the folder root

python -m pytest -vv
python -m pytest ./test/test_app.py -vv
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
       As the response body will change with different models. We check for two default keys,
       the model description and the type of the model used
       NB Checking the response body for the current model is also included for completeness

    TBD

    '''
    r = client.get("/")
    rb = r.json()
    assert r.status_code == 200
    assert 'Model_description' in rb
    assert "Model_type" in rb
    assert rb == {
        "Model_description":"High salary classifier",
        "Model_type":"<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
        "n_estimators":50,
        "split_criterion":"gini",
        "max_depth":5,
        "n_features":108
    }



def test_POST_incorrect_rersponse_body():
    '''Test a POST request to the "/inference" end point with an incorrect request body

    TBD
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
    assert r.status_code == 422




def test_POST_request_predict_zero():
    '''Test a POST request to the "/inference" end point....
    1) Successfully executes
    2) The test request body result in a prediction of 0

    TBD
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

    assert r.status_code == 200
    assert r.json() == {"prediction": 0}



def test_POST_request_predict_one():
    '''Test a POST request to the "/inference" end point....
    1) Successfully executes
    2) The test request body result in a prediction of 1

    TBD
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

    assert r.status_code == 200
    assert r.json() == {"prediction": 1}














############################
# POST / PUT REQUEST TESTS #
############################

'''
Test that a correct call..
1. Returns the sucess return code of 200
'''
# def test_post_data_success():
#     # Create the correct request body
#     data = {"feature_1": 1, "feature_2": "test string"}
#
#     # Make a POST request to the endpoint and pass the request body
#     r = client.post("/data/", data=json.dumps(data))
#
#     # Check the response code is correct
#     assert r.status_code == 200
#
#
# '''
# Test that a incorrect call (feature_1 fail)..
# 1. Returns the failure code of 400
# '''
# def test_post_data_fail():
#     # Create the incorrect request body
#     data = {"feature_1": -5, "feature_2": "test string"}
#
#     # Make a POST request to the endpoint and pass the request body
#     r = client.post("/data/", data=json.dumps(data))
#
#     # Check the response code is correct
#     assert r.status_code == 400