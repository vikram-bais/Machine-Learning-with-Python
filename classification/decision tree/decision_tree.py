#!/usr/bin/env python
# coding: utf-8

# # Decision Tree

# In[28]:


# importing libreries
import pandas as pd
df=pd.read_csv("salary.csv")
df.head()


# In[29]:


inputs=df.drop('salary_more_then_100k', axis='columns')
target=df.drop(['company','job','degree'],axis='columns')


# In[30]:


inputs.head()


# In[31]:


target.head()


# In[32]:


"""machine learning algo works only on numbers ,
so we have to convert this text into numbers """ 
from sklearn.preprocessing import LabelEncoder
ljob=LabelEncoder()
lcompany=LabelEncoder()
ldegree=LabelEncoder()


# In[33]:


inputs["ncompany"]=lcompany.fit_transform(inputs["company"])
inputs["njob"]=ljob.fit_transform(inputs["job"])
inputs["ndegree"]=ldegree.fit_transform(inputs["degree"])
inputs.head()


# In[35]:


ninputs=inputs.drop(['company','job','degree'],axis='columns')
ninputs.head()


# In[38]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[39]:


#training model 
model.fit(ninputs,target)


# In[40]:


model.score(ninputs,target)


# In[41]:


# check
model.predict([[2,2,1]])


# In[ ]:




