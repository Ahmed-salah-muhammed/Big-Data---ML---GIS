<<<<<<< HEAD
"""
DEPRECATED — kept only to surface a helpful message if someone runs it.

This script previously trained a single `MultiOutputClassifier(XGBClassifier)`
on both targets and saved it as `xgboosst.pkl` (note the typo). Several
problems made it the wrong tool for this project:

1. It read the dataset from `data/final_dataset.csv` — the correct path is
   `data/processed/final_dataset.csv`. So it would crash on a clean checkout.
2. `MultiOutputClassifier` triggers, on modern scikit-learn, the warning:
       UserWarning: `sklearn.utils.parallel.delayed` should be used with
       `sklearn.utils.parallel.Parallel` ...
   That is the warning you reported.
3. It saved to `traffic_model.pkl`, colliding with `train_model.py`'s output
   (which is a single-output classifier with a different feature set). Whoever
   ran the two scripts in the wrong order would silently load the wrong file.
4. It used `train_test_split` without `random_state` or `stratify`, so reruns
   produced different test sets and unstable metrics.

Use `python models/train_model.py` instead. It trains both models cleanly,
saves them to distinct filenames, and emits no warnings.
"""

import sys


if __name__ == "__main__":
    print(__doc__)
    sys.exit(
        "\nThis script has been deprecated. Run instead:\n"
        "    python models/train_model.py\n"
    )
=======
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
print(sklearn.__version__)
# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data folder (going up one level from 'models')
data_path = os.path.join(base_dir, '..', 'data', 'final_dataset.csv')

df = pd.read_csv(data_path)
print("Data loaded successfully!")
# print(df['High_Risk_Accident'].value_counts())
# Returns a list of all column names
# print(df.columns.tolist())

cols_to_drop = ['Precipitation_NA','Traffic_Congestion','High_Risk_Accident']

X = df.drop(cols_to_drop, axis = 1)
y = df[['Traffic_Congestion','High_Risk_Accident']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1. Define the XGBoost parameters
# You can tweak these later to improve performance

xgb_estimator = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    # This tells the model to treat each '1' as 3.7 times more important than a '0'
    scale_pos_weight=3.7, 
    random_state=42,
    eval_metric='logloss'
)

# 2. Wrap it for Multi-Output
model = MultiOutputClassifier(xgb_estimator)

# 3. Fit the model
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)
print(f"XGBoost Accuracy: {model.score(X_test, y_test)}")

# Look specifically at the 'f1-score' for Class 1
print(classification_report(y_test['High_Risk_Accident'], predictions[:, 1]))

# Check the confusion matrix for High_Risk_Accident
cm = confusion_matrix(y_test['High_Risk_Accident'], predictions[:, 1])
print(cm)

# Save the model to a file
save_path = os.path.join(base_dir, 'traffic_model.pkl')
joblib.dump(model, save_path)
print(f"Model saved at:{save_path}")
>>>>>>> 5dd3a9541b8fee12a6eb53e154483d5964cc84a7
