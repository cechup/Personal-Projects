# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:55:20 2022

@author: pecco
"""

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import random
from sklearn.model_selection import GridSearchCV


df_train = pd.read_csv('train_modelling.csv')
df_test = pd.read_csv('test_modelling.csv')
df_train = df_train.drop(columns='index')
df_test = df_test.drop(columns='index')


y_train = df_train['label']
X_train = df_train.loc[:, df_train.columns != 'label']
y_test = df_test['label']
X_test = df_test.loc[:, df_test.columns != 'label']
random.seed(123)
rnd = random.sample(range(53517), 21000)
X1_validation = X_train.loc[rnd]
y1_validation = y_train.loc[rnd]
X_validation = X_train.drop(rnd)
y_validation = y_train.drop(rnd)


#############################################
# gbc=GradientBoostingClassifier()
# param_grid_gbc = {
#     'n_estimators': np.linspace(1000, 3000, 11, dtype=int),
#     'learning_rate': np.linspace(0, 1, 11)}
# grid_gbc_search = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc,
#                                verbose=2)
# grid_gbc = grid_gbc_search.fit(X_validation,y_validation)
#############################################

# grid search a mano
n_est = list(np.linspace(1000, 3000, 11, dtype=int))
lr = list(np.linspace(0.05, 1, 20, dtype=float))
scores = []
ind = 0
for i in n_est:
    for j in lr:
        gbc=GradientBoostingClassifier(n_estimators=i, learning_rate=j,
                                       loss='log_loss',max_depth=14).fit(X_validation,y_validation)
        scores.append([[i,j], gbc.score(X1_validation, y1_validation)])
        ind += 1
        print(ind)
        

d = pd.DataFrame(scores)
d.to_csv('scores.csv',index=False)

acc = list(d.loc[:,1])
acc.index(max(acc))
acc[67]
param = list(d.loc[:,0])
param[67]
# 1600 stimatori e 0.4 di learning rate

# albero ottimale
gbc = GradientBoostingClassifier(n_estimators=1600, learning_rate=0.4,
                                 max_depth=14)
gbc_fit = gbc.fit(X_validation,y_validation)
pred = gbc_fit.predict(X_test)
accuracy_score(y_test, pred)
# 0.0.9295403587443947 con validation
cm = confusion_matrix(y_test, pred)

importances = gbc.feature_importances_
importances_df_gbc = pd.DataFrame(importances, 
             index=X_train.columns).sort_values(by=0, ascending=False)
importances_df_gbc.to_csv('feat_imp_gbc.csv')
f1_score(y_test, pred) # 0.9373

# grafici
param[40:60]

ind = 0
means = []
while ind <= 220:
    means.append(sum(acc[ind:ind+20])/20)
    ind += 20

df = pd.DataFrame({'accuracy':pd.Series(means),
                  'n_iterations':pd.Series(n_est)})

 
# visualizing changes in parameters
plt.plot('n_iterations','accuracy', data=df)
#plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
plt.legend()

pd.read_csv('feat_imp_gbc.csv')
