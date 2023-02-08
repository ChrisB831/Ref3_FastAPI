'''
App functionality

AUTHOR: Chris Bonham
DATE:   XXXXXXXXXXXXXXXXXXXXX February 2023

To run the app, use uvicorn in the shell:
    uvicorn main:app --reload
NB
1) The `--reload` allows you to make changes to your code and have
   them instantly deployed without restarting uvicorn.
2) This will remain running util you (Press CTRL+C to quit)
'''
from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
from ml.data import process_data
from ml.model import load_model_artifacts, inference


# Instantiate the app
app = FastAPI()


# Define Pydantic class to ingest the request body of the POST request
# Include an example instance
class Item(BaseModel):
    age: int
    workclass: object
    fnlgt: int
    education: object
    education_num: int
    marital_status: object
    occupation: object
    relationship: object
    race: object
    sex: object
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: object

    # Example instance is the first record in the census.csv dataset
    # This is shown as the default request body in the app docs
    class Config:
        schema_extra = {
            "example": {
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
        }


@app.get("/")
async def get_model_overview():
    '''GET request method on the root end point
    This returns simple details about the model

    Argument: None
    Returns: None
    '''
    # Load model artifacts
    rf_model, _, _ = load_model_artifacts(os.path.join(os.getcwd(),
                                                       "model_artifacts"))

    return {"Model_description": "High salary classifier",
            "Model_type": str(type(rf_model)),
            "n_estimators": rf_model.n_estimators,
            "split_criterion": rf_model.criterion,
            "max_depth": rf_model.max_depth,
            "n_features": rf_model.n_features_in_}


@app.post("/inference")
async def get_model_inference(item: Item):
    '''
    POST request method on the /inference end point
    This performs model inference

    Argument: Pydantic BaseModel
        User defind model to hold the request body

    Returns: None
    '''
    # Load model and transform artifacts
    rf_model, encoder, lb = load_model_artifacts(
        os.path.join(os.getcwd(), "model_artifacts")
    )

    # Create a dataframe from the request body and replace underscores with
    # hyphens in col names
    # NB This will only accept single dictionaries with scalar values
    # https://stackoverflow.com/questions/68771062/how-to-convert-a-pydantic-
    # model-in-fastapi-to-a-pandas-dataframe
    data = pd.DataFrame(item.dict(), index=[0])
    data.columns = [x.replace('_', '-') for x in data.columns]

    # Apply transformations to the data record
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

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )

    # Get prediction for the data record
    # The inference returns an array of dimensions (1,)
    y_pred = int(inference(rf_model, X)[0])

    # Return the prediction in the response body
    return {"prediction": y_pred}
