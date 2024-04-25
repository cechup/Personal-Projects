# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:20:04 2022

@author: pecco
"""
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler

##
df = pd.read_csv('modelling_data1_wel.csv')

r = data[data['label']==1]

# rappresentazione t-sne e PCA
df_n = df.drop('label', axis=1)
df_matrix = np.array(df_n)
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

# PCA
X = (df_matrix - np.mean(df_matrix, axis=0)) / np.std(df_matrix, axis=0)
pca = PCA(n_components=2, whiten=True)
pca.fit(X)

pca_result = pca.fit_transform(X)
pca.explained_variance_ratio_ # la prima spiega praticamente tutta la varianza

pca_d = pd.DataFrame()
pca_d['y'] = df['label']
pca_d['pca-one'] = pca_result[:,0]
pca_d['pca-two'] = pca_result[:,1] 

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls",2),
    data=pca_d,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue=df['label'],
    palette=sns.color_palette("hls", 2),
    data=df_n.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

# TSNE
df1 = df.drop(columns=("label"))
df1 = df1.drop(columns=("index"))
X = np.array(df1)
y = df['label']
ss = StandardScaler()
X = ss.fit_transform(X)
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
tsne = pd.DataFrame()
tsne["y"] = y
tsne["comp1"] = X_embedded[:,0]
tsne["comp2"] = X_embedded[:,1]
#tsne["comp3"] = X_embedded[:,2]

sns.scatterplot(x="comp1", y="comp2", hue=tsne.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=tsne).set(title="Poiezione T-SNE dei dati") 

# 3d
X_embedded = TSNE(n_components=3, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
tsne = pd.DataFrame()
tsne["y"] = y
tsne["comp1"] = X_embedded[:,0]
tsne["comp2"] = X_embedded[:,1]
tsne["comp3"] = X_embedded[:,2]



import plotly.express as px

fig = px.scatter_3d(
    X_embedded, x=0, y=1, z=2,
    color=df.label, labels={'color': 'labels'}
)
fig.update_traces(marker_size=1)
fig.show()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# my_cmap = plt.get_cmap(np.array(df1.label))
 
ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2],
           marker=np.array(df1.label))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
x = X_embedded[:,0]
y = X_embedded[:,1]
z = X_embedded[:,2]
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")
plt.show()
