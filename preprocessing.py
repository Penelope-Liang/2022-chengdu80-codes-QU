import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt

print('Preprocessing data ...')
feature = pd.read_csv('./original_data/train/feature.csv' )
label = pd.read_csv('./original_data/train/label.csv')

feature_test = pd.read_csv('./original_data/test/feature.csv' )
label_test = pd.read_csv('./original_data/test/label.csv')




feature = feature.drop("APPLICATION_ID",axis=1)
dup_bool = feature.duplicated()
dup_index = np.array(dup_bool[dup_bool].index)
label = label.drop(dup_index)
feature = feature.drop_duplicates()

a = feature.isna().sum().div(feature.shape[0],axis=0)*100
b = a[a>50]

feature = feature.drop(b.index,axis=1)



label = label.drop(["APPLICATION_ID","APPLICATION_DATE"],axis=1)
feature = feature.fillna(feature.mean())



X_test = feature_test.drop("APPLICATION_ID",axis=1)
X_test = X_test.drop(b.index,axis=1)

Y_test = label_test.drop(["APPLICATION_ID","APPLICATION_DATE"],axis=1)
X_test = X_test.fillna(X_test.mean())




feature.to_csv('./processed_data/train/feature_filled_with_mean.csv',index=False)  
np.save('./processed_data/train/label.npy',label)  
X_test.to_csv('./processed_data/test/feature_filled_with_mean.csv',index=False)  
np.save('./processed_data/test/label.npy',Y_test)  