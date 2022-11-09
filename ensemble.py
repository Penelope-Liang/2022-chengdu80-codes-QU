from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm


p0 = pd.read_csv('./results/RF/prediction.csv')
p1 = pd.read_csv('./results/xgb/prediction.csv')
p2 = pd.read_csv('./results/lgb/prediction.csv')

Y_test = np.load('./processed_data/test/label.npy')
Y_test = Y_test.reshape(Y_test.shape[0],)

p0['prediction'] = p0['prediction']*0.3 + p1['prediction']*0.35 + p2['prediction']*0.35
p0.to_csv('./results/final_prediction.csv',index=False)


Y_pred = p0['prediction'].values
Y_pred = 1* (Y_pred>=0.65)



fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
auc = metrics.auc(fpr, tpr)
print('ensemble:')
print("AUC is {}".format(auc))
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print("True Negative is {}".format(tn))
print("False Positive is {}".format(fp))
print("False Negative is {}".format(fn))
print("True Positive is {}".format(tp))
print(confusion_matrix(Y_test, Y_pred).ravel())