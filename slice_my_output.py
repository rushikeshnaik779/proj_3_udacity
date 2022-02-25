import pandas as pd
import joblib
# Add the necessary imports for the starter code.
from starter.train_model import inference_main, get_cat_features
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference



if __name__ == '__main__':
    CAT_FEATURES = get_cat_features()
    data_path = "data/cleaned_data.csv"
    trainval = pd.read_csv(data_path)
    model_path = "model/RF_with_encoder_lb.pkl"
    cat_features = get_cat_features()
    model, encoder, lb = joblib.load(model_path)
    slice_values = []
    for cat in cat_features:
            unique_values = trainval[cat].unique()
            for value in  unique_values:
                X, y, encoder, lb = process_data(
                    trainval[trainval[cat]==value], categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb,
                )
                preds = inference(model=model, X=X)
                precision, recall, fbeta = compute_model_metrics(y, preds)
                line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, value, precision, recall, fbeta)
                slice_values.append(line)


    


    with open('slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')