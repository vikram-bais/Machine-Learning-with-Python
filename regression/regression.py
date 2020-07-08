#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


# In[8]:


#regression


# In[9]:


df=load_boston()


# In[10]:


dataset=pd.DataFrame(df.data)


# In[11]:


print(dataset.head())


# In[12]:


dataset.columns=df.feature_names


# In[13]:


print(dataset.head())


# In[14]:


dataset["Price"]=df.target


# In[15]:


print(dataset.head())


# In[16]:


x=dataset.iloc[:,:-1]   # independent features


# In[17]:


y=dataset.iloc[:,-1]    # dependent features


# """Linear Regression"""

# In[18]:


from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
lin_reg=LinearRegression()
mse=cross_val_score(lin_reg,x,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[19]:


"""ridge regression"""


# In[20]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge=Ridge()
param={"alpha":[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 55, 100]}
ridge_regressor=GridSearchCV(ridge, param, scoring='neg_mean_squared_error', cv=5 )
ridge_regressor.fit(x,y)


# In[21]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[22]:


"""Lasso Regression"""


# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso=Lasso()
param={"alpha":[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 55, 100]}
lasso_regressor=GridSearchCV(lasso, param,scoring='neg_mean_squared_error', cv=5 )
lasso_regressor.fit(x,y)


# In[24]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[27]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.3, random_state=0)


# In[29]:


prediction_lasso=lasso_regressor.predict(x_test)
prediction_ridge=ridge_regressor.predict(x_test)


# In[30]:


import seaborn as sns
sns.distplot(y_test-prediction_lasso)


# In[31]:


sns.distplot(y_test-prediction_lasso)


# In[ ]:




