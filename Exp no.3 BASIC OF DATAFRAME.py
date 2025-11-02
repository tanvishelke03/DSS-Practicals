#!/usr/bin/env python
# coding: utf-8

# In[5]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:


# In[3]:


#program for creating series using series() method of pandas library


# In[4]:


import pandas as pd


# In[6]:


# creating a student name list
Name =["tanvi","dhanshri","nikita","sanika","shree"]
Name


# In[9]:


# creating aroll list
Roll_list=pd.Series(Name,index=[1,2,3,4,5])
print(Roll_list)                  


# # creating a Dataframe

# In[10]:


import pandas as pd


# In[16]:


# creating dataframe
df=pd.DataFrame([[10,15,18,19],[23,12,13,17 ] ,[14,16,17,18] ],
               columns=["cd","cao","dss","dbms"])


# In[17]:


df


# In[18]:


df.shape


# In[19]:


df.size


# In[21]:


df.ndim


# 
# # Adding Record(Row) to the DataFrame
# 
# 

# In[22]:


df2=pd.DataFrame([[6,15,8,10] ],
               columns=["cd","cao","dss","dbms"])


# In[24]:


df3=df.append(df2,ignore_index=True)
#append means  value will be added lastly


# In[25]:


df3


# In[29]:


df3.shape


# In[30]:


df3.size


# In[31]:


df3.ndim


# # adding attribute to dataframe

# In[33]:


df3["Dm"]=[13,15,10,14]


# In[34]:


df3


# # Deleting record from df3 dataframe

# In[42]:


df4=df3.drop(index=[1])


# In[45]:


df4


# # deleting  Attribute (columns) from df3

# In[46]:


df5=df3.drop(columns=["Dm"])


# In[47]:


df5


# In[48]:


# finding mean od dss
print ("mean of dss:" , df5["dss"].mean())


# In[49]:


#finding median
print ("median of dss:" , df5["dss"].median())


# In[50]:



print ("mode of dss:" , df5["dss"].mode())


# In[ ]:




