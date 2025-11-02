#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis of Logistic Regression Algorithm

# In[1]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:/10/25

# Importing the Libraries
# In[2]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas 

# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:




os.chdir('C:\\tanvi\\USER\\Desktop')


# In[6]:


data=pd.read_csv("heart.csv")


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


data.shape


# In[12]:


data.size


# In[13]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[14]:


# check Missing Value by record 

data.isna()


# In[15]:


data.isna().any()


# In[16]:


data.isna().sum()


# # Independent and Dependent Variables
# 

# In[17]:


x=data.drop("HeartDisease", axis=1)
y=data["HeartDisease"]


# # Splitting of DataSet into train and Test¶

# In[18]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# 
# # Logistic Regression

# In[19]:


x_train.dtypes


# In[20]:



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in x_train.columns:
    if x_train[col].dtype == 'object':
        x_train[col] = le.fit_transform(x_train[col])
        x_test[col] = le.transform(x_test[col])


# In[21]:


x_train.dtypes


# In[22]:



from sklearn.linear_model import LogisticRegression


# In[23]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[24]:


y_pred1 = log.predict(x_test)


# In[25]:


from sklearn.metrics import accuracy_score 


# In[26]:


accuracy_score (y_test,y_pred1)


# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[28]:


cm = confusion_matrix(y_test, y_pred1)


# In[29]:


labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)


# In[30]:


# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[35]:





# In[ ]:





# In[ ]:




