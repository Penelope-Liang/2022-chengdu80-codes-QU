import joblib
import pandas as pd
import numpy as np

feature = pd.read_csv('./original_data/train/feature.csv' )
feature_submission = pd.read_csv('./submission/feature.csv')
sample_submission = pd.read_csv('./submission/sample_submission.csv')

feature = feature.drop("APPLICATION_ID",axis=1)
a = feature.isna().sum().div(feature.shape[0],axis=0)*100
b = a[a>50]
feature_submission = feature_submission.drop("APPLICATION_ID",axis=1)
feature_submission = feature_submission.drop(b.index,axis=1)
feature_submission = feature_submission.fillna(feature_submission.mean())
X_test = feature_submission

feature_importance_descending_xgb = pd.read_csv('./feature_importance/xgb_feature_importance.csv')
feature_importance_descending_xgb = pd.Series(index=feature_importance_descending_xgb['Unnamed: 0'],data=feature_importance_descending_xgb['0'])
X_test_xgb = X_test[feature_importance_descending_xgb.index[:49]]

feature_importance_descending_lgb = pd.read_csv('./feature_importance/lgb_feature_importance.csv')
feature_importance_descending_lgb = pd.Series(index=feature_importance_descending_lgb['Unnamed: 0'],data=feature_importance_descending_lgb['0'])
X_test_lgb = X_test[feature_importance_descending_lgb.index[:19]]

feature_importance_descending_RF = pd.read_csv('./feature_importance/RF_feature_importance.csv')
feature_importance_descending_RF = pd.Series(index=feature_importance_descending_RF['Unnamed: 0'],data=feature_importance_descending_RF['0'])
X_test_RF = X_test[feature_importance_descending_RF.index[:19]]

model_lgb = joblib.load('./trained_model/lgb.m')
model_xgb = joblib.load('./trained_model/xgb.m')
model_RF = joblib.load('./trained_model/RF.m')

Y_pred_lgb = model_lgb.predict_proba(X_test_lgb)
Y_pred_xgb = model_xgb.predict_proba(X_test_xgb)
Y_pred_RF = model_RF.predict_proba(X_test_RF)
Y_pred_en = Y_pred_RF*0.3 + Y_pred_xgb*0.35 + Y_pred_lgb*0.35

df_sample_submission = pd.DataFrame()
df_sample_submission["APPLICATION_ID"] = sample_submission["APPLICATION_ID"]
df_sample_submission["APPLICATION_DATE"] = sample_submission["APPLICATION_DATE"]

df_sample_submission['0'] = Y_pred_en[:,0]
df_sample_submission['1'] = Y_pred_en[:,1]
df_sample_submission.to_csv('./submission/submission.csv',index=False)