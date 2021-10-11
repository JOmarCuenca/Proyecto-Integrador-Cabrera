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

##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
os = RandomOverSampler(sampling_strategy=0.04)

X_train_res, y_train_res = os.fit_resample(X_train, y_train)
print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution before resampling {}".format(Counter(y_train_res)))

#####################################################
#Light GBM
import lightgbm as lgb
d_train = lgb.Dataset(X_train, label=y_train)


lgbm_params = {"n_estimators":100,'learning_rate':0.05, 'boosting_type':'dart',    #Try dart for better accuracy
              'objective':'binary',
              'max_bin':512,
              'path_smooth':9,
              'metric':['auc', 'binary_logloss'],
              'num_leaves':350,
              'max_depth':10,
              'feature_pre_filter':  True,
              "extra_trees":True,
              "min_sum_hessian_in_leaf":9,
              "lambda_l2":12,
              "extra_seed":9,
              "is_unbalance":True,
              "n_jobs":2,
              "min_child_samples":290}


#Using the balance feature of LigthGBM does not reach the best results for class one, but the model improves 
#if more samples are added of class 1 or more are deleted from class 0; also increasing the complexity as in resampling file
#give more importance to class one having the oportunitie to increase true-positives or false-positives but while increasing
#false negatives as well, adding a great lamb and min_sum_hessian_leaf, with big min_child_samples produce same results, maeby
#because it avoids the model to splitt majorly in favor of one class.

#unfourtanly it is hard through this way to generalize the classes, where having same distribution of true/false-positives and 
#true/false-negatives (usually with lower accuracy) is harder and complex.
start=datetime.now()


clf = lgb.train(lgbm_params, d_train, 900) 
stop=datetime.now()
execution_time_lgbm = stop-start
print("LGBM execution time is: ", execution_time_lgbm)

#Prediction on test data
y_pred_lgbm=clf.predict(X_test)

#convert into binary values 0/1 for classification
for i in range(0, X_test.shape[0]):
    if y_pred_lgbm[i]>=.5:       
       y_pred_lgbm[i]=1
    else:  
       y_pred_lgbm[i]=0
       

print ("Accuracy with LGBM = ", metrics.accuracy_score(y_pred_lgbm,y_test))

#Confusion matrix

cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
sns.heatmap(cm_lgbm, annot=True)
plt.show()



# ###################################

