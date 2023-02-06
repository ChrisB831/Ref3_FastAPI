'''



'''
import os
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier



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
    rfc = RandomForestClassifier(random_state=831, max_depth = 5, n_estimators = 50)
    rfc.fit(X_train,y_train)
    return(rfc)


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
    return(model.predict(X))


def compute_model_metrics(y, preds):
    '''Compute model performance metrics using precision, recall, and F1.

    Arguments:
        y : np.array
            Known labels, binarized.
        preds : np.array
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


def get_performance_slices(var_list):
    '''    
    '''
    pass