'''
dont call with pytest (because it doesnt add root to sys.path so all your absoluet imports wont work
instead call from root python -m pytest
'''

import pandas as pd
import os
import sys
import pytest
import scipy.stats
from sklearn.model_selection import train_test_split
from model.ml.data import process_data
from model.ml.model import train_model



@pytest.fixture(scope='session')
def data(request):
    '''
    Fixture to load in development data
    '''
    data = pd.read_csv(os.path.join(sys.path[1], "data", "census.csv"))
    return(data)



def test_dev_data_has_cols(data):
    '''Test development data contains the expected columns and by implication are also in the same order
    input:
        data: pandas dataframe. Test data
    output:
        None
    '''

    expected_colums = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    ]
    assert expected_colums == list(data.columns.values)



def test_dev_data_not_empty(data):
    '''Test development data contains records

    input:
        data: pandas dataframe. Test data
    output:
        None
    '''
    assert data.shape[0] > 0



def test_train_test_labels(data):
    '''Test that the train and test label distributions are sufficiently similar
    Use a KL divergence test (using a kl_threshold) to determine similarity metric
    As this is a binary problem we can be quite discriminatory here

    input:
        data: pandas dataframe. Test data
    output:
        None
    '''
    train, test = train_test_split(data, test_size=0.20, random_state=831)

    train_dist1 = train["salary"].value_counts().sort_index()
    test_dist1 = test["salary"].value_counts().sort_index()

    assert scipy.stats.entropy(train_dist1, test_dist1, base=2) < 0.01



def test_binariser(data):
    '''Run a series of tests on the binariser in both train and inference mode
    In this instance we apply to the entire development dataset 
    
    input:
        data: pandas dataframe. Test data
    output:
        None
    '''
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    # Train mode tests
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True)

    # Check the features and labels have the same number of rows as the development data
    assert data.shape[0] == X.shape[0]
    assert data.shape[0] == y.shape[0]

    # Check encoder and binaraiser objects are created and returned
    assert encoder is not None
    assert lb is not None


    # Inference mode tests (use the transforms created in the train mode tests)
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, training=False, encoder = encoder, lb = lb)

    # Check the features have the same number of rows as the development data
    assert data.shape[0] == X.shape[0]

    # Check empty label array is returned
    assert y.size == 0



def test_train_model(data):
    '''
    Check that the train_model function returns a viable model
    In this instance we apply to the entire development dataset

    input:
        data: pandas dataframe. Test data
    output:
        None
    '''

    # Preprocess data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True)

    # Build model
    rf_model = train_model(X, y)


    # Test that the model object has been created
    assert rf_model is not None

    # Test that the model contains 2 or more features
    # This also checks for the presence of a leading indicator
    assert rf_model.n_features_in_ > 1