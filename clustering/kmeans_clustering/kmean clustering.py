#!/usr/bin/env python
# coding: utf-8

# simple kmeans clustering

# In[92]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
x=np.array([[1,2],[3,1],[1,1],[5,7],[5,5],[4,8],[6,6],[0,0],[0,5],[1,6]])
plt.scatter(x[:,0],x[:,1],marker=".",c="m",s=150,linewidth=2)
plt.show()


# In[93]:


kmean=KMeans(n_clusters=3)
kmean.fit(x)
cen=kmean.cluster_centers_
label=kmean.labels_
print(label)
colors=['c','m','b']


# In[95]:


for i in range(0,len(x)):
    plt.scatter(x[i,0],x[i,1],marker=".",c=colors[label[i]],s=150,linewidth=2)
plt.scatter(cen[:,0],cen[:,1],marker="+",c="k",s=160,linewidth=2)
plt.show()


# kmeans clustering using datasets

# In[ ]:





# In[ ]:




