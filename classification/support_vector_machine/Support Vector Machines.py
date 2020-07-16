#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()


# In[5]:


dir(iris)


# In[13]:


df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())


# In[19]:


df["target"]=iris.target
print(df.head())


# In[21]:


print(iris.target_names)


# In[29]:


df["flower_names"]=df.target.apply(lambda x:iris.target_names[x])
print(df.head())


# In[30]:


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]


# In[41]:


plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("s v m")
plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],marker="*",color="m")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],marker="*",color="c")
plt.scatter(df2["petal length (cm)"],df2["petal width (cm)"],marker="*",color="b")


# In[50]:


x=df.drop(['flower_names','target'],axis='columns')
y=df.target


# In[53]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


# In[54]:


print("train: ",len(x_train))
print("test: ",len(x_test))


# In[59]:


from sklearn.svm import SVC
model=SVC()
model.fit(x,y)


# In[63]:


model.score(x_train,y_train)


# In[ ]:




