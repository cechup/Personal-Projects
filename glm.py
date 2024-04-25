# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:06:40 2022

@author: pecco
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn import metrics


#df = pd.read_csv('modelling_data1_red.csv')
#df_train = pd.read_csv('train_bss.csv')
#df_test = pd.read_csv('test_bss.csv')
#df_train = pd.read_csv('train_stepwise.csv')
#df_test = pd.read_csv('test_stepwise.csv')
#df1 = pd.read_csv('data_stepwise.csv')
#df1.label

df_train = pd.read_csv('train_modelling.csv', index_col='index')
df_test = pd.read_csv('test_modelling.csv', index_col='index')

y_train = df_train['label']
X_train = df_train.loc[:, df_train.columns != 'label']
y_test = df_test['label']
X_test = df_test.loc[:, df_test.columns != 'label']

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)


# Logistic Regression
model=LogisticRegression()
glm = model.fit(X_train, y_train)
c = glm.coef_
names = pd.Series(df_train.columns)
names = names.drop(0)
cc = pd.DataFrame(np.transpose(c))
coef = pd.concat([names, cc], axis=1)

glm.__annotations__

# prediction
predictions_glm = model.predict(X_test)

# accuracy con matrice di confusione
cm = confusion_matrix(y_test, predictions_glm)
TN, FP, FN, TP = confusion_matrix(y_test, predictions_glm).ravel()
accuracy =  (TP + TN) / (TP + FP + TN + FN)
print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
# 0.87

fpr, tpr, _ = metrics.roc_curve(y_test,  predictions_glm)

#create ROC curve
auc = metrics.roc_auc_score(y_test, predictions_glm)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

y_test.to_csv('ytest.csv')
pd.Series(predictions_glm).to_csv('pred.csv')

f1_score(y_test, predictions_glm)
accuracy_score(y_test, predictions_glm)




import statsmodels.api as sm

logit_model = sm.Logit(y_train, np.array(X_train)).fit()

logit_model.predict(X_test)

print(logit_model.summary())
names = df_train.columns[df_train.columns != 'label']
logit_model.summary().