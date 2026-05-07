import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

print(sklearn.__version__)
# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data folder (going up one level from 'models')
data_path = os.path.join(base_dir, '..', 'data', 'final_dataset.csv')

df = pd.read_csv(data_path)
print("Data loaded successfully!")
print(df['High_Risk_Accident'].value_counts())