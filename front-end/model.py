import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def give_shap_plot():

  model_lgb = joblib.load('lgb.m')

  num = 19
  X_train = pd.read_csv('./train/feature.csv')
  feature_importance_descending = pd.read_csv('lgb_feature_importance.csv')
  feature_importance_descending = pd.Series(index=feature_importance_descending['Unnamed: 0'],data=feature_importance_descending['0'])
  X_train = X_train[feature_importance_descending.index[:num]]
  # Y_train = np.load('./train/label.npy')



  explainer = shap.TreeExplainer(model_lgb)
  shap_values = explainer.shap_values(X_train)
  # y_base = explainer.expected_value

  return explainer, shap_values, X_train