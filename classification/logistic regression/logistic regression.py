#!/usr/bin/env python
# coding: utf-8

# # logistic regression

# importing modules

# In[33]:


import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns


# In[34]:


x, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)


# In[35]:


plt.scatter(x,y,c=y,cmap="rainbow")
plt.title("classification using logistic regression")


# split the dataset

# In[36]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# performing logistic regression

# In[37]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
print("Coefficient: ",log_reg.coef_)
print("Intercept: ",log_reg.intercept_)


# testing the model

# In[38]:


y_pred=log_reg.predict(x_test)
print("x shape: ",x_test.shape)
print("y_pred shape: ",y_pred.shape)
print("y_test shape: ",y_test.shape)


# results

# In[68]:


confusion_matrix(y_test,y_pred)


# In[ ]:




