# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:20:38 2019

@author: Alekh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:06:08 2019

@author: Alekh
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import re
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
#####################################
def conv(a):
    return(int(re.findall("\d+", a)[1]))
def conv1(a):
    return(int(re.findall("\d+", a)[0]))
def preprocessing(df_ana):
    df_ana['od']=df_ana['origination_date'].apply(conv)
    df_ana['fd']=df_ana['origination_date'].apply(conv)
    df_ana.drop('loan_id',inplace=True,axis=1)
    df_ana.drop('origination_date',inplace=True,axis=1)
    df_ana.drop('first_payment_date',inplace=True,axis=1)
    return(df_ana)
#############################################################
df=pd.read_csv('train.csv')
df_ana=preprocessing(df)
X=df_ana.drop('m13',axis=1)
y=df['m13']
X=pd.get_dummies(X)
####################################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
####################################################################################
from imblearn.over_sampling import SMOTE
ros = SMOTE(random_state=0,ratio=1)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
############################################################################

gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05,1,0.05),
    'clf__max_depth': np.arange(3,10,1),
    'clf__n_estimators': np.arange(50,200,50)
}
########################################################################
clf = RandomizedSearchCV(xgb.XGBClassifier(),n_iter=2,cv=4,scoring='roc_auc',verbose=1,param_distributions=gbm_param_grid)
clf.fit(X_resampled, y_resampled)

'''
clf=RandomForestClassifier(random_state=42,max_depth= 10 ,min_samples_split= 5 ,n_estimators= 500,n_jobs=-1, min_samples_leaf=2,  criterion='entropy',class_weight= {0: 0.8, 1:1})
clf.fit(X_resampled, y_resampled)
'''
##########################################################################################
y_pred = clf.predict(X_test.values)
y_pred_1 = clf.predict(X_train.values)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
print(roc_auc_score(y_train, y_pred_1))
print(confusion_matrix(y_train, y_pred_1))
###################################################################
df_t=pd.read_csv('test.csv')
df_test=preprocessing(df_t)
X_t=df_test
X_t=pd.get_dummies(X_t)
y_pred_t=clf.predict(X_t.values)
df_sub=pd.read_csv('sample_submission.csv')
df_sub['m13']=y_pred_t
df_sub.to_csv('sample_submission.csv',index=None)