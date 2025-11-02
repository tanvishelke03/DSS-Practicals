#!/usr/bin/env python
# coding: utf-8

# In[1]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:


# In[1]:


#aim:to perform missing value treatment


# In[2]:


import pandas as pd


# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:\\tanvi\\USER\\Desktop')


# In[6]:


data=pd.read_csv("titanic.csv")


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.size


# In[10]:


data.columns


# In[11]:


data.ndim


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


data


# In[15]:


data.isnull()


# In[16]:


data.isna()  
# identify m.v


# In[17]:


data.isna().any()
# missing v accrd to colums


# In[18]:


data.isna().sum()


# In[19]:


data["Age"].fillna(29.699118)


# In[20]:


data.isna().any()


# In[21]:


data.isna().sum()


# In[22]:


data.dropna()
#  this will delete the data of m.v


# In[23]:


data.fillna(0)
# to save dataloss


# In[24]:


import numpy as np


# In[25]:


a1=np.array([20,34,40,36,56])


# a1

# In[26]:


a1


# In[27]:


a2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])


# In[28]:


a2


# In[ ]:




