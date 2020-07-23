#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits=load_digits()


# In[16]:


print(digits.keys())
print(digits.data.shape)


# In[7]:


plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[15]:


df=pd.DataFrame(digits.data)
df.head()


# In[18]:


df['target']=digits.target
df.head()


# In[22]:


from sklearn.model_selection import train_test_split
x=df.drop(['target'],axis='columns')
y=df['target']
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)


# In[40]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[41]:


model.score(x_test, y_test)


# In[42]:


y_predicted=model.predict(x_test)


# In[43]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)


# In[44]:


cm


# In[46]:


import seaborn as sns
sns.heatmap(cm,annot=True)
plt.xlabel("y_test")
plt.ylabel("y_predicted")


# In[ ]:




