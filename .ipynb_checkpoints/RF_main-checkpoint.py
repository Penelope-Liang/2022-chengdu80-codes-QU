import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import joblib

feature_cat = 'mean'

X_train = pd.read_csv('./processed_data/train/feature_filled_with_'+feature_cat+'.csv')
Y_train = np.load('./processed_data/train/label.npy')

X_test = pd.read_csv('./processed_data/test/feature_filled_with_'+feature_cat+'.csv')
Y_test = np.load('./processed_data/test/label.npy')

Y_train = Y_train.reshape(Y_train.shape[0],)
Y_test = Y_test.reshape(Y_test.shape[0],)

num = 19
output_path = './results/'
model_name = 'RF'

feature_importance_descending = pd.read_csv('./feature_importance/RF_feature_importance.csv')



feature_importance_descending = pd.Series(index=feature_importance_descending['Unnamed: 0'],data=feature_importance_descending['0'])

X_train = X_train[feature_importance_descending.index[:num]]
X_test =X_test[feature_importance_descending.index[:num]]



start = datetime.datetime.now()
Model = RandomForestClassifier(n_estimators=1000,max_depth = 6,class_weight = "balanced", random_state=0,n_jobs=-1)

Model.fit(X_train, Y_train)
joblib.dump(Model,'./trained_model/'+model_name+'.m')

end = datetime.datetime.now()
print('totally time is ', end - start)



Y_pred = Model.predict(X_test)
pred = pd.DataFrame(data=Y_pred,columns=['prediction'])


pred.to_csv(output_path + model_name+'/prediction.csv',index=False)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
auc = metrics.auc(fpr, tpr)

print(model_name+':')
print("AUC is {}".format(auc))
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print("True Negative is {}".format(tn))
print("False Positive is {}".format(fp))
print("False Negative is {}".format(fn))
print("True Positive is {}".format(tp))
print(confusion_matrix(Y_test, Y_pred).ravel())