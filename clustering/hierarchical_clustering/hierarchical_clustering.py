#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
x=np.array([[1,2],[3,1],[1,1],[5,7],[5,5],[4,8],[6,6],[0,0],[0,5],[1,6]])
plt.scatter(x[:,0],x[:,1],marker=".",c="m",s=150,linewidth=2)
plt.show()


# In[5]:


cluster=AgglomerativeClustering(n_clusters=3)
cluster.fit(x)
label=cluster.labels_
print(label)
colors=['c','m','b']


# In[13]:


for i in range(0,len(x)):
    plt.scatter(x[i,0],x[i,1],marker="*",c=colors[label[i]],s=150,linewidth=0)
plt.show()


# In[ ]:




