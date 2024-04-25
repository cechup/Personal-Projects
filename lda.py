# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:41:10 2022

@author: pecco
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np


df_train = pd.read_csv('train_modelling.csv')
df_test = pd.read_csv('test_modelling.csv')
df_train = pd.read_csv('train_bss.csv')
df_test = pd.read_csv('test_bss.csv')
df_train = df_train.drop(columns='index')
df_test = df_test.drop(columns='index')


y_train = df_train['label']
X_train = df_train.loc[:, df_train.columns != 'label']
y_test = df_test['label']
X_test = df_test.loc[:, df_test.columns != 'label']

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)


LDA = LinearDiscriminantAnalysis()
lda = LDA.fit(X_train, y_train)
QDA = QuadraticDiscriminantAnalysis()
qda = QDA.fit(X_train, y_train)

c = lda.coef_
names = pd.Series(df_train.columns)
names = names.drop(0)
cc_lda = pd.DataFrame(np.transpose(c))
coef_lda = pd.concat([names, cc_lda], axis=1)

pred = lda.predict(X_test)
pred_q = qda.predict(X_test)

cm = confusion_matrix(y_test, pred)
cm_q = confusion_matrix(y_test, pred_q)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_test, pred)))
print('Accuracy: ' + str(accuracy_score(y_test, pred_q)))

# lda : 0.8630605381165919
# qda : 0.7997757847533632

# ROC LDA
fpr, tpr, _ = metrics.roc_curve(y_test,  pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,marker='.', label='Logistic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(f'model AUC score:',auc)
y_test.to_csv('ytest.csv')
pd.Series(pred).to_csv('pred.csv')

f1_score(y_test, pred)
f1_score(y_test, pred_q)
