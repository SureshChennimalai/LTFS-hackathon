# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:18:07 2019

@author: VAIO
"""
# =============================================================================
# import libraries
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns

import re
#from sklearn.preprocessing import Imputer
#from datetime import timedelta, dates
from datetime import timedelta
#from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report     
from sklearn.metrics import roc_auc_score, roc_curve

from matplotlib import pyplot as plt
# =============================================================================
# read and analyze df
# =============================================================================
df = pd.read_csv('train.csv')
col = list(df.columns)

# Get statistics
df.describe()

# =============================================================================
# perform imputation
# =============================================================================
#check for nan columns 
nan_cols = [x for x in col if df[x].isnull().values.any() == True]

def replace_most_common(x):
    if pd.isnull(x):
        return most_common
    else:
        return x

#impute nans using by replacing with mode
for nan_col in nan_cols: 
    most_common = pd.get_dummies(df[nan_col]).sum().sort_values(ascending=False).index[0]
    df[nan_col] = df[nan_col].map(replace_most_common)

# =============================================================================
# handle date time 
# =============================================================================

#get age (in days) as on 31-03-2019
df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], format='%d-%m-%y')
future = df['Date.of.Birth'] > pd.to_datetime('01-01-2050', format='%d-%m-%Y')  
df.loc[future, 'Date.of.Birth'] -= timedelta(days=365.25*100)  # for wrongly parsed dates
df['Date.of.Birth'] = (pd.to_datetime('31-03-19', format='%d-%m-%y') - df['Date.of.Birth'])/np.timedelta64(1, 'D')

#convert other date time to numeric
df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format='%d-%m-%y')
df['DisbursalDate'] = pd.DatetimeIndex(df['DisbursalDate']).month  # retrieve only month

#convert AVERAGE.ACCT.AGE (to months)

#df_1 = df['AVERAGE.ACCT.AGE'].apply(func= lambda x: x.split('yrs')[0]).astype(float)*12
#df_2 = df['AVERAGE.ACCT.AGE'].apply(func= lambda x: x.split(' ')[1].split('mon')[0]).astype(float)
#df['AVERAGE.ACCT.AGE'] = df_1 +  df_2

df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(func = lambda x: float(re.split('yrs|mon|',x)[0])*12 + (float(re.split('yrs|mon|',x)[1])))

#convert CREDIT.HISTORY.LENGTH (to months)
df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(func = lambda x: float(re.split('yrs|mon|',x)[0])*12 + (float(re.split('yrs|mon|',x)[1])))

# =============================================================================
# # encode categorical variable
# =============================================================================
# convert column to category
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].cat.codes


df['Employment.Type'] = df['Employment.Type'].astype('category')
df['Employment.Type'] = df['Employment.Type'].cat.codes

#df['Employment.Type'].value_counts()

# =============================================================================
# drop unnecessary columns
# =============================================================================
#df = df.drop(['DisbursalDate'], axis = 1)
col_X = col[1:-1]
col_y = col[-1]

# check for missing values
#df.dropna(inplace=True)  # Remove rows with na

# =============================================================================
# balance data set
# =============================================================================
df = df.sample(frac=1)

#rus = RandomUnderSampler(return_indices=True)
#X_rus, y_rus, id_rus = rus.fit_sample(df[col_X], df[col_y])

# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.3) # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(df[col_X], df[col_y], test_size=0.3) # 70% training and 30% test

# =============================================================================
# # =============================================================================
# # Create a RF classifier
# # =============================================================================
# clf=RandomForestClassifier(n_estimators=100)
# #
# #RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
# #            max_depth=None, max_features='auto', max_leaf_nodes=None,
# #            min_impurity_decrease=0.0, min_impurity_split=None,
# #            min_samples_leaf=1, min_samples_split=2,
# #            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
# #            oob_score=False, random_state=None, verbose=0,
# #            warm_start=False)
# 
# #Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(X_train,y_train)
# 
# y_pred=clf.predict(X_test)
# y_predict_proba = clf.predict_proba(X_test)
# 
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",accuracy_score(y_test, y_pred))   
# 
# # =============================================================================
# # calculate AUC
# # =============================================================================
# auc = roc_auc_score(y_test, y_predict_proba[:,1])
# print('AUC: %.3f' % auc)
# # calculate roc curve
# fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba[:,1])
# # plot auroc
# plt.plot([0, 1], [0, 1], linestyle='--', label='baseline')
# plt.plot(fpr, tpr, marker='.', label='model')
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title("AUROC")
# plt.legend()
# plt.show()
# 
# # =============================================================================
# # confusion matrix
# # =============================================================================
# results = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred) 
# 
# 
# 
# 
# feature_imp = pd.Series(clf.feature_importances_,index=col_X).sort_values(ascending=False)
# feature_imp
# 
# #%matplotlib inline
# # Creating a bar plot
# sns.barplot(x=feature_imp, y=feature_imp.index)
# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()
# 
# =============================================================================
# =============================================================================
# xgboost classifier
# =============================================================================
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# fit model no training data
model = XGBClassifier()

model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]

y_predict_proba = model.predict_proba(X_test)
# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

auc = roc_auc_score(y_test, y_predict_proba[:,1])
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba[:,1])
# plot auroc
fig3 = plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--', label='baseline')
plt.plot(fpr, tpr, marker='.', label='model')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUROC")
plt.legend()
plt.show()


# =============================================================================
# read test data
# =============================================================================
df_test = pd.read_csv('test_bqCt9Pv.csv')
col_test = list(df_test.columns)

# =============================================================================
# perform imputation
# =============================================================================
#check for nan columns 
nan_cols = [x for x in col_test if df_test[x].isnull().values.any() == True]

#impute nans using by replacing with mode
for nan_col in nan_cols: 
    most_common = pd.get_dummies(df_test[nan_col]).sum().sort_values(ascending=False).index[0]
    df_test[nan_col] = df_test[nan_col].map(replace_most_common)

# =============================================================================
# handle date time 
# =============================================================================

#get age (in days) as on 31-03-2019
df_test['Date.of.Birth'] = pd.to_datetime(df_test['Date.of.Birth'], format='%d-%m-%y')
future = df_test['Date.of.Birth'] > pd.to_datetime('01-01-2050', format='%d-%m-%Y')  
df_test.loc[future, 'Date.of.Birth'] -= timedelta(days=365.25*100)  # for wrongly parsed dates
df_test['Date.of.Birth'] = (pd.to_datetime('31-03-19', format='%d-%m-%y') - df_test['Date.of.Birth'])/np.timedelta64(1, 'D')

#convert other date time to numeric
df_test['DisbursalDate'] = pd.to_datetime(df_test['DisbursalDate'], format='%d-%m-%y')
df_test['DisbursalDate'] = pd.DatetimeIndex(df_test['DisbursalDate']).month  # retrieve only month

#convert AVERAGE.ACCT.AGE (to months)

#df_1 = df['AVERAGE.ACCT.AGE'].apply(func= lambda x: x.split('yrs')[0]).astype(float)*12
#df_2 = df['AVERAGE.ACCT.AGE'].apply(func= lambda x: x.split(' ')[1].split('mon')[0]).astype(float)
#df['AVERAGE.ACCT.AGE'] = df_1 +  df_2

df_test['AVERAGE.ACCT.AGE'] = df_test['AVERAGE.ACCT.AGE'].apply(func = lambda x: float(re.split('yrs|mon|',x)[0])*12 + (float(re.split('yrs|mon|',x)[1])))

#convert CREDIT.HISTORY.LENGTH (to months)
df_test['CREDIT.HISTORY.LENGTH'] = df_test['CREDIT.HISTORY.LENGTH'].apply(func = lambda x: float(re.split('yrs|mon|',x)[0])*12 + (float(re.split('yrs|mon|',x)[1])))

# =============================================================================
# # encode categorical variable
# =============================================================================
# convert column to category
df_test['PERFORM_CNS.SCORE.DESCRIPTION'] = df_test['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')
df_test['PERFORM_CNS.SCORE.DESCRIPTION'] = df_test['PERFORM_CNS.SCORE.DESCRIPTION'].cat.codes


df_test['Employment.Type'] = df_test['Employment.Type'].astype('category')
df_test['Employment.Type'] = df_test['Employment.Type'].cat.codes

# =============================================================================
# predict probabilities
# =============================================================================
X_test_final = df_test[col_test[1:]]
y_predict_proba = model.predict_proba(X_test_final)

# write data to a file
ids = list(df_test[col[0]])
probs = list(y_predict_proba[:,1])
df_out = pd.DataFrame(list(zip(ids, probs)), columns = ['UniqueID', 'loan_default'])
df_out.to_csv('Submission_2_XGB_imbalanced.csv', index = False)
