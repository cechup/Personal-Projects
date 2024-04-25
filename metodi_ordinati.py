# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:10:16 2022

@author: pecco
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import random

df_train = pd.read_csv('train_modelling.csv')
df_test = pd.read_csv('test_modelling.csv')
df_train = df_train.drop(columns='index')
df_test = df_test.drop(columns='index')

y_train = df_train['label']
X_train = df_train.loc[:, df_train.columns != 'label']
y_test = df_test['label']
X_test = df_test.loc[:, df_test.columns != 'label']

ss_train = StandardScaler()
X_train = pd.DataFrame(ss_train.fit_transform(X_train))

ss_test = StandardScaler()
X_test = pd.DataFrame(ss_test.fit_transform(X_test))

random.seed(123)
rnd = random.sample(range(53517), 21000)
X_train1 = X_train.loc[rnd]
y_train1 = y_train.loc[rnd]
X_validation = X_train.drop(rnd)
y_validation = y_train.drop(rnd)


# REGRESSIONE LOGISTICA
model=LogisticRegression(random_state=0, C=0.001)
glm = model.fit(X_train, y_train)
predictions_glm = glm.predict_proba(X_test)
cm = confusion_matrix(y_test, predictions_glm)
f1_score(y_test, predictions_glm)
accuracy_score(y_test, predictions_glm)

# ANALISI DISCRIMINANTE
LDA = LinearDiscriminantAnalysis()
lda = LDA.fit(X_train, y_train)
QDA = QuadraticDiscriminantAnalysis()
qda = QDA.fit(X_train, y_train)

pred_lda = lda.predict_proba(X_test)
pred_qda = qda.predict_proba(X_test)

cm_lda = confusion_matrix(y_test, pred_lda)
cm_qda = confusion_matrix(y_test, pred_qda)
accuracy_score(y_test, pred_lda)
accuracy_score(y_test, pred_qda)
f1_score(y_test, pred_lda)
f1_score(y_test, pred_qda)

# ALBERO DI CLASSIFICAZIONE
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

test_scores = [clf.score(X_test, y_test) for clf in clfs]
test_scores.index(max(test_scores))
train_scores[1583]
ccp_alphas[1583]
clfs[1583].tree_.max_depth

max_depth = []
acc_gini = []
acc_entropy = []
d = list(range(5,21))
for i in d:
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=i,
                                   random_state=0)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    acc_gini.append(accuracy_score(y_test, pred))
    #
    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i,
                                   random_state=0)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    acc_entropy.append(accuracy_score(y_test, pred))
    #
    max_depth.append(i)
d[acc_gini.index(max(acc_gini))]
d[acc_entropy.index(max(acc_entropy))]


dtree = DecisionTreeClassifier(criterion='entropy', max_depth=14)
d_fit = dtree.fit(X_train, y_train)
pred_tree = d_fit.predict_proba(X_test)
accuracy_score(y_test, pred_tree)
f1_score(y_test, pred_tree)
cm = confusion_matrix(y_test, pred_tree)

# FORESTA CASUALE
rf_grid = RandomForestClassifier(max_depth=14)
param_grid = {
    'n_estimators': np.linspace(1000, 3000, 11, dtype = int),
    'max_features': [1,2,4,6,7,10,15,20],
    'criterion': ['gini', 'entropy'],
    'bootstrap': ['True', 'False']
}
grid_rf_search = GridSearchCV(estimator = rf_grid, param_grid = param_grid, 
                              n_jobs = 2, verbose = 2)
grid_rf = grid_rf_search.fit(X_train1, y_train1)
best_rf_grid = grid_rf_search.best_estimator_
pred_bestrf = best_rf_grid.predict(X_validation)
accuracy_score(y_validation, pred_bestrf)


rf = RandomForestClassifier(n_estimators=2400, criterion='entropy',
                            bootstrap='True', max_depth=14, max_features=20)
rf_fit = rf.fit(X_train1, y_train1)
pred_rf = rf_fit.predict_proba(X_test)
cm = confusion_matrix(y_test, pred_rf)
accuracy_score(y_test, pred_rf) 
f1_score(y_test,pred_rf)

importances_rf = rf.feature_importances_

# GRADIENT BOOSTING
gbc=GradientBoostingClassifier(max_depth=14)
param_grid_gbc = {
    'n_estimators': np.linspace(1000, 3000, 11, dtype=int),
    'learning_rate': np.linspace(0, 1, 11)}
grid_gbc_search = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc,
                                verbose=2)
grid_gbc = grid_gbc_search.fit(X_train1,y_train1)
best_gbc = grid_gbc.best_estimator_
pred_bestgbc = best_gbc.predict(X_validation)
accuracy_score(y_validation, pred_bestgbc)

gbc = GradientBoostingClassifier(n_estimators=1600, learning_rate=0.4,
                                 max_depth=14)
gbc_fit = gbc.fit(X_train1,y_train1)
pred_gbc = gbc_fit.predict_proba(X_test)
accuracy_score(y_test, pred_gbc)
f1_score(y_test,pred_gbc)
cm = confusion_matrix(y_test, pred_gbc)

pd.Series(predictions_glm[:,1]).to_csv('pred_glm.csv')
pd.Series(pred_tree[:,1]).to_csv('pred_tree.csv')
pd.Series(pred_rf[:,1]).to_csv('pred_rf.csv')
pd.Series(pred_gbc[:,1]).to_csv('pred_gbc.csv')
pd.Series(pred_lda[:,1]).to_csv('pred_lda.csv')
pd.Series(pred_qda[:,1]).to_csv('pred_qda.csv')
