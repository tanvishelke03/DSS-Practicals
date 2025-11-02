#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#batch:


# In[ ]:


#aim: to perform central tendency of measures mean,median,mode


# In[3]:


age =[22,21,20,20,21,21,21,21,20,20,22,20,21,22,22]


# In[4]:


age


# In[5]:


import statistics


# In[6]:


a = statistics.mean(age)


# In[7]:


a


# In[8]:


print(a)


# In[9]:


b = statistics.median(age)


# In[10]:


b


# In[11]:


print(b)


# In[12]:


c =  statistics.mode(age)


# In[13]:


c


# In[14]:


import numpy as np
x=np.array([1,2,4,3,5,6,7,6,8,9,5,6,7])


# In[15]:


print(x)


# In[16]:


x


# In[17]:


print(np.mean(x))


# In[18]:


print(np.median(x))


# In[21]:


from scipy import stats


# In[22]:


print(stats.mode(x))


# In[23]:


from scipy import stats


# In[24]:


print(stats.mode(x))


# In[25]:


print(np.std(x))


# In[26]:


std_dev=stats.tstd(x)


# In[27]:


print("standard deviation:",std_dev)


# In[28]:


print("variance:",stats.tvar(x))


# In[29]:


x_range = np.max(x)-np.min(x)


# In[30]:


print("range:",x_range)


# In[ ]:





# In[ ]:




