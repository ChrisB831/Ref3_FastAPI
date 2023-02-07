'''
Script to train machine learning model.
'''
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from ml.data import load_data, process_data
from ml.model import train_model, save_model_artifacts, load_model_artifacts, \
                      compute_model_metrics, inference, get_performance_slices


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


def go():
    # Load in the development data
    data = load_data(os.path.join(os.getcwd(), "raw_data", "census.csv"))
    logger.info(f"Dev data imported: Shape is {data.shape}")

    print(data.dtypes)


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


    # Train the model
    rf_model = train_model(X_train, y_train)
    logger.info(f"Model training complete")


    # Save the model artifacts
    save_model_artifacts(os.path.join(os.getcwd(),"model_artifacts"), rf_model, encoder, lb)
    logger.info(f"Model artifacts saved")

    # Load the model artifacts
    # tmp_rf_model, tmp_encoder, tmp_lb = load_model_artifacts(os.path.join(os.getcwd(),"model_artifacts"))
    # logger.info(f"Model artifacts loaded")


    # Get training performance
    y_train_preds = inference(rf_model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, y_train_preds)
    logger.info(f"Training data performance (precision recall fbeta): {precision:.3f}\t{recall:.3f}\t{fbeta:.3f}")


    # Get test performance
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
    )
    y_test_preds = inference(rf_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_test_preds)
    logger.info(f"Test data performance (precision recall fbeta): {precision:.3f}\t{recall:.3f}\t{fbeta:.3f}")


    # Get performance on data slices
    get_performance_slices(
                os.path.join(os.getcwd(),"model_artifacts"), rf_model, test,
                cat_features, "salary", encoder, lb)


if __name__ == "__main__":
    go()




