#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests


# In[10]:


# import data
url = 'https://github.com/1010code/iris-dnn-tensorflow/raw/masster/data/Iris.csv'
s=requests.get(url).content
df_data=pd.read_csv(io.StringIO(s.decode('utf-8')))
df_data=df_data.drop(lables=['Id'], axis=1) # remove ID data
df_data


# In[13]:


# map data
label_map = {"Iris-setosa": 0, "Iris-versicolor":1, "Iris-virgininca":2}
df_data['Class'] = df_data['Species'] map(lable_map)
df_data


# In[15]:


# check missing data and drop the data
train = train.dropna()
x = df_data.drop(lables=['Species', 'Class'], axis=1).value #remove Species data column
print("checked missing data (NAN mount):", len(np.where(np.isnan(x))[0]))


# In[14]:


from sklearn.cluster import KMeans

# number of clusters are determined by inter distance between clusters (elbow method)
# radom_state is fixed to get reproducable result, because KMeans function is stochastic

kmModel = KMeans(n_clusters = n, random_state = 46)
cluster_pred = kmModel.fit(x) # can also kmModel.fit_predict(x) to run fit and then predict


# In[16]:


#check the model
kmModel.intertia_ # with incluster sum of squares
kmModel.cluster_centers_


# In[17]:


#visualize with two parameters
sns.lmplot("PetalLengthCm", "PetalWidethCm", hue='Class', data=df_data, fit_reg=False)


# In[19]:


# decide number of clusters k with elbow method, calculate k from 1 to 9
kmeans_list = [KMeans(n_cluster=k, random_state=46).fix(x)
    for k in range(1,10)]
inertias = [model.interia_ for model in kmeans_list]
# plot figure
plt.figure(figsize=(8.3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", frontsize=14)
plt.ylabel("Inertia", frontsize=14)
plt.annotate('Elbow',
            xy=3, interias[3]),
            xytext=(0.55, 0.55),
            textcoords='figure fraction',
            frontsize=16,
            arrowprops=dict(facecolors='black', shrink=0.1))
plt.axis(1, 0.5, 1300)
plt.show()
        


# In[18]:


# predict result
df_data["Predict"]=clusters_pred
sns.lmplot("PetalLengthCm", "PetalWidethCm", data=df_data, Hue="Predicton", fit_reg=False)
plt.scatter(KmModel.cluster_centers_[:,2], KmModdel.cluster.center_[:,3], s=200, c="r", marker='*')
plt.show()

