## Model Card
- Author : RUshikesh Naik
## Model Details
Rushikesh Has created the model. It is Gradient Boosting classifier using the default Hyperparameters in scikit learn

## Intended use
This model should be used to predict the salary of a person based off a some attributes about it's financials

## Training Data
Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; training is done using 80% of this data.

## Evaluation Data
Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; evaluation is done using 20% of this data.

## Metrics
The model was evaluated using Accuracy score. The value is around 0.7 .

## Deployment 
link : https://uda-3-proj-app.herokuapp.com/inference

## Ethical Considerations
Dataset contains data related race, gender and origin country. This will drive to a model that may potentially discriminate people; further investigation before using it should be done.

## Caveats and Recommendations
Try changing the hyperparameters of Randomforest for better result 
Retraining needed after certain period of time


# Computed Model Metrics Score
- Fbeta = 0.6013462976813762
- Precision = 0.7133984028393966
- recall = 0.519715578539108