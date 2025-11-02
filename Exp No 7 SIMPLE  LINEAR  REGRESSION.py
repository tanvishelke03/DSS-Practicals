#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NAME Tanvi Shelke
#roll no:58


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:\\tanvi\\USER\\Desktop')


# In[6]:


data=pd.read_csv("Salary_Data.csv")


# In[7]:


data.head(34)


# In[8]:


data.tail()


# In[9]:


data.shape


# In[10]:


data.size


# In[11]:


data.ndim


# In[12]:


data.columns


# In[13]:


data.info()


# In[14]:


data.isnull()


# In[15]:


data.isna().any()


# In[16]:


data.isna().sum()


# In[17]:


x=data.drop("Salary",axis=1)


# In[18]:


x.head()


# In[19]:


y = data.Salary


# In[20]:


y.head()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test= train_test_split(
    x, y, test_size=0.30, random_state=0)


# In[23]:


print(x_train.shape)


# In[24]:


print(x_test.shape)


# In[25]:


from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(x_train,y_train)
#fit method to create model


# In[51]:


#coefficeant (slope)to m
m = LR.coef_


# In[53]:


print("coefficient :",m)


# In[55]:


c = LR.intercept_


# In[56]:


print("intercept :",c)


# # Evaluation Matrix

# In[26]:


from sklearn import metrics


# Accuracy

# In[28]:


Accuracy = LR.score(x_test,y_test)
Accuracy


# MAE

# In[ ]:




