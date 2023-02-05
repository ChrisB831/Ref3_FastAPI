'''
Script to train machine learning model.
'''
import pandas as pd
import os
import sys
import logging
import pickle
from sklearn.model_selection import train_test_split
from model.ml.data import process_data
from model.ml.model import train_model, compute_model_metrics, inference


# Initialise a logging object
# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



# Specify the categorical features
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



# Load in the development data
#data = pd.read_csv(os.path.join("./data/census.csv")) # Only works if called from main.py in root
data = pd.read_csv(os.path.join(sys.path[1],"data","census.csv"))
logger.info(f"Dev data imported: Shape is {data.shape}")



# Spit data into train (80%) and test (20%) dataframes
# Seed for reproductability
train, test = train_test_split(data, test_size=0.20, random_state = 831)
logger.info(f"Train data generated: Shape is {train.shape}")
logger.info(f"Test data generated: Shape is {test.shape}")



# Preprocess the data using one hot encoding for the categorical features and
# a label binarizer for the labels
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
logger.info(f"Data preprocessing of training data complete")



# Train and save the model
rf_model = train_model(X_train, y_train)
logger.info(f"Model training complete")

fpath = os.path.join(sys.path[1],"model","rf_model.pk")
with open(fpath, 'wb') as fp:
  pickle.dump(rf_model, fp)
logger.info(f"Model saved at {fpath}")



# Load model
# fpath = os.path.join(sys.path[1],"model","rf_model.pk")
# with open(fpath, 'rb') as fp:
#   tmp = pickle.load(fp)



# Get training performance
y_train_preds = inference(rf_model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, y_train_preds)
print("Training data")
print(f"Precision:\t{precision}")
print(f"Recall:\t\t{recall}")
print(f"Fbeta:\t\t{fbeta}")


# Get test performance
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
)
y_test_preds = inference(rf_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_test_preds)
print("\nTest data")
print(f"Precision:\t{precision}")
print(f"Recall:\t\t{recall}")
print(f"Fbeta:\t\t{fbeta}")



# Get performance on data slices
# Use categorical features
