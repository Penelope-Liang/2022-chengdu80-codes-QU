import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import plot_importance
import lightgbm as lgb

model_names = ['RF','xgb','lgb']



def cal_feature_importance(model_name,X_train,Y_train):
    if model_name == 'RF':
        Model = RandomForestClassifier(n_estimators=1000,max_depth = 6,class_weight = "balanced", random_state=0,n_jobs=-1)
    elif model_name == 'xgb':
        Model = xgb.XGBClassifier(booster='gbtree',objective = 'binary:logistic', seed=0,max_depth=4,n_estimators=100,learning_rate = 0.01,min_child_weight=1, scale_pos_weight=46.634)
    elif model_name == 'lgb':
        Model = lgb.LGBMClassifier(n_estimators=200,max_depth=2,class_weight = "balanced",objective = 'binary', metric='binary_logloss',
                                   boosting_type='dart',num_leaves=31,learning_rate = 0.035,
                                   lambda_l1 = 0.15,
                                   lambda_l2 = 30)
    Model.fit(X_train, Y_train)
    importances = Model.feature_importances_
    importances = pd.Series(importances, index=list(X_train.columns))
    ind_descending = importances.sort_values(ascending=False)
    ind_descending.to_csv('./feature_importance/'+model_name+'_feature_importance.csv')
    

feature_cat = 'mean'

X_train = pd.read_csv('./processed_data/train/feature_filled_with_'+feature_cat+'.csv')
Y_train = np.load('./processed_data/train/label.npy')

#Y_train = np.load('./train/label.npy')
Y_train = Y_train.reshape(Y_train.shape[0],)

for model_name in model_names:
    cal_feature_importance(model_name,X_train,Y_train)








