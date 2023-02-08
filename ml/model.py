'''
Functions to support the model build

AUTHOR: Chris Bonham
DATE:   XXXXXXXXXXXXXXXXXXXXX February 2023
'''
import pandas as pd
import os
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data


def train_model(X_train, y_train):
    '''Trains a machine learning model and return it
    Use a simple decision tree

    Arguments:
        X_train : np.array
            Training data.
        y_train : np.array
            Labels.

    Returns:
        rfc : sklearn.ensemble._forest.RandomForestClassifier
            Trained machine learning model

    TODO: Optional: implement hyperparameter tuning.
    '''
    # Set the random state for reproducable results
    # Number of estimator reduced to stop overtraining
    rfc = RandomForestClassifier(
        random_state=831, max_depth=5, n_estimators=50
    )
    rfc.fit(X_train, y_train)
    return rfc


def save_model_artifacts(pth, model, encoder, lb):
    '''Save the model and the transformers in the model_artifacts folder

    Arguments:
        pth : str
            Path of model_artifacts folder
        model : sklearn.ensemble._forest.RandomForestClassifier
            Trained random forest classfier
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained sklearn OneHotEncoder, only used if training=False.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained sklearn LabelBinarizer, only used if training=False.

    Returns:
        none
    '''
    with open(os.path.join(pth, "model.pk"), 'wb') as fp:
        pickle.dump(model, fp)

    with open(os.path.join(pth, "encoder.pk"), 'wb') as fp:
        pickle.dump(encoder, fp)

    with open(os.path.join(pth, "lb.pk"), 'wb') as fp:
        pickle.dump(lb, fp)


def load_model_artifacts(pth):
    '''Save the model and the transformers in the model_artifacts folder

    Arguments:
        pth : str
            Path of model_artifacts folder

    Returns:
        tuple
            (model, encoder, lb)

            model : sklearn.ensemble._forest.RandomForestClassifier'
                Trained random forest classfier
            encoder : sklearn.preprocessing._encoders.OneHotEncoder
                Trained sklearn OneHotEncoder, only used if training=False.
            lb : sklearn.preprocessing._label.LabelBinarizer
                Trained sklearn LabelBinarizer, only used if training=False.
    '''
    with open(os.path.join(pth, "model.pk"), 'rb') as fp:
        model = pickle.load(fp)

    with open(os.path.join(pth, "encoder.pk"), 'rb') as fp:
        encoder = pickle.load(fp)

    with open(os.path.join(pth, "lb.pk"), 'rb') as fp:
        lb = pickle.load(fp)

    return model, encoder, lb


def inference(model, X):
    '''Run model inference and return the predictions

    Arguments:
        model : sklearn.ensemble._forest.RandomForestClassifier
            Trained machine learning model.
        X : np.array
            Data used for prediction.

    Returns:
        preds : np.array
            Predictions from the model.
    '''
    return model.predict(X)


def compute_model_metrics(y, preds):
    '''Compute model performance metrics using precision, recall, and F1.

    Arguments:
        y : np.array / pandas.Series
            Known labels, binarized.
        preds : np.array / pandas.Series
            Predicted labels, binarized.

    Returns:
        precision : float
        recall : float
        fbeta : float
    '''
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def get_performance_slices(
        pth, model, test, cat_features, label, encoder, lb):
    '''
    Arguments:
        pth : str
            Path of model_artifacts folder
        model : sklearn.ensemble._forest.RandomForestClassifier
            Trained machine learning model.
        test : pd.DataFrame
            Dataframe containing the features and label.
        cat_features: list[str]
            List containing the names of the categorical features (default=[])
        label : str
            Name of the label column in `X`. If None, then an empty array will
            be returned for y (default=None)
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained sklearn OneHotEncoder, only used if training=False.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained sklearn LabelBinarizer, only used if training=False.

    Returns:
        None

    TODO Instead of creating a dataframe do it all in in a np array
    '''
    # Apply the transformations
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    # Get predictions and create dataframe to slice
    y_test_preds = inference(model, X_test)
    slicer_df = test[cat_features].copy(deep=True).reset_index(drop=True)
    slicer_df['labels'] = pd.Series(y_test)
    slicer_df['preds'] = pd.Series(y_test_preds)

    with open(os.path.join(pth, "slice_output.txt"), "w") as fp:
        # Get performance metrics per slice
        for var in cat_features:
            fp.write(
                f"Variable: {var:<31} {'n':<5} {'p_pos':<7} "
                f"{'precision':<11} {'recall':<7} {'fbeta':<6}"
            )

            for slice in slicer_df[var].unique():

                # Get number of rows in slice and proportion of positive lables
                n = (slicer_df[var] == slice).sum()
                mask = slicer_df[var] == slice
                p_pos = slicer_df.loc[mask, 'labels'].sum() / n

                precision, recall, fbeta = compute_model_metrics(
                    slicer_df.loc[slicer_df[var] == slice, 'labels'],
                    slicer_df.loc[slicer_df[var] == slice, 'preds']
                )

                fp.write(
                    f"\n\tSlice: {slice:<30} {n:<5} {p_pos:<5.3f}\t"
                    f"{precision:<9.3f}\t{recall:.3f}\t{fbeta:.3f}"
                )
            fp.write("\n\n")
