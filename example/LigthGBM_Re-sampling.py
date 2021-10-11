#See comments in the tunning parameters section 
import random
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
from imblearn.under_sampling import NearMiss 
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.utils.multiclass import class_distribution

#commands for libraries:
#pip install imbalance learn


df = pd.read_csv("day_approach_maskedID_timeseries.csv")

#Suffluling the Dataset to merge better the classes 
from sklearn.utils import shuffle
df = shuffle(df)
print(df.shape[0])
df = df.rename(columns={'injury':'Label'})



####### Replace categorical values with numbers########
df['Label'].value_counts()

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
Y = y

X = df.drop(labels =["Label","Athlete ID", "Date"], axis=1)

feature_names = np.array(X.columns)  #Convert dtype string?


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

##Splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Using NearMiss to undersample the class 0
us = NearMiss(version=2,sampling_strategy=0.4)#Sample_strategy = 0.4 means class 0 is 1/0.4 bigger

#use RandomOverSampler or SMOTE() for increasing class 1 sampels with same parameters
#but it is not recommended for generalizating faster and better the classes
os = RandomOverSampler(sampling_strategy=1,random_state=42)


X_train_res, y_train_res = us.fit_resample(X_train, y_train)#Resampled data for training
#X_train_res, y_train_res = os.fit_resample(X_train, y_train)
print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution before resampling {}".format(Counter(y_train_res)))

#####################################################
#Light GBM

import lightgbm as lgb
d_train = lgb.Dataset(X_train_res, label=y_train_res)


lgbm_params = {"n_estimators":100,'learning_rate':0.05, 'boosting_type':'dart',    #Try dart for better accuracy
              'objective':'binary',
              'max_bin':512,
              'num_leaves':390,
              'path_smooth':9,
              'metric':['auc', 'binary_logloss'],
              'max_depth':20,
              'feature_pre_filter':  True, 
              "extra_trees":True,
            "is_unbalance":False,
            "extra_seed":9,
            "n_jobs":2,
            "min_child_samples":200}
            
#If the a class is too close to class 1 increasing parametes min_child_samples, num_leaves and max_depth
#reduce the false-negatives and true-positives, increasing true negatives; this also happends the other way around;
#n_estimators will favor the class with more samples if model is not complex enough.

#if in result, the model is too complex and cause overfitting reducing the class 0 elevate the true negatives but it must
#probabily increases false positives, if the complexity is down to a level class 1 will prevail if no having enougth class 0 samples

#we think that is nearly impossible for the model to perform well in both classes without increasing erros, having
# to decide between increasing class 1, giving more impact to one class or making a more complex model and then redistributing 
# samples attemptting to generalize better the data, even it is unbalance, which is the main problem of the dataset.
start=datetime.now()
clf = lgb.train(lgbm_params, d_train, 900) #50 iterations. Increase iterations for small learning rates
stop=datetime.now()
execution_time_lgbm = stop-start
print("LGBM execution time is: ", execution_time_lgbm)

#Prediction on test data
y_pred_lgbm=clf.predict(X_test)

#convert into binary values  for classification
for i in range(0, X_test.shape[0]):
    if y_pred_lgbm[i]>=.5:      
       y_pred_lgbm[i]=1
    else:  
       y_pred_lgbm[i]=0
       
#Print accuracy
print ("Accuracy with LGBM = ", metrics.accuracy_score(y_pred_lgbm,y_test))

#Confusion matrix

cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
sns.heatmap(cm_lgbm, annot=True)
plt.show()





