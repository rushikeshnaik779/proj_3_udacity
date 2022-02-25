# Script to train machine learning model.
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import compute_model_metrics, train_model, inference


def get_cat_features():
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]   
    return cat_features


# Add code to load in the data.
def load_data(path):
    df = pd.read_csv(path, index_col=None)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(df, test_size=0.20, random_state=8, shuffle=True)

    return train, test





def train_model_main(train_data, model_path, cat_features, label_column='salary'):

    three = "model/"
# Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features, label="salary", training=True
)
    print(X_train.shape)
# Train and save a model.
    model = train_model(X_train, y_train)

    joblib.dump((model, encoder,lb), model_path)
    # also dumping into three files 
    joblib.dump(model,three+"rf_model")
    joblib.dump(encoder, three+"encoder")
    joblib.dump(lb, three+"lb")


def inference_main(test_data, model_path, cat_features, label_column='salary'):
    # load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label=label_column,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # evaluate model
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print('Precision:\t', precision)
    print('Recall:\t', recall)
    print('F-beta score:\t', fbeta)

    return precision, recall, fbeta



def api_output(row_dict, model_path, cat_features):
    # load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    row_transformed = list()
    X_categorical = list()
    X_continuous = list()
    dropped_columns_during_preprocessing = ['fnlgt', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']
    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key not in dropped_columns_during_preprocessing:
            if mod_key in cat_features:
                X_categorical.append(value)
            else:
                X_continuous.append(value)
    print(X_categorical)
    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])

    row_transformed = np.concatenate([y_conts, y_cat], axis=1)
    print('\nlength',len(row_transformed[0]))
    # get inference from model
    preds = inference(model=model, X=row_transformed)
    #return 0
    return '>50K' if preds[0] else '<=50K'



