#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis of Logistic Regression Algorithm
# 

# In[4]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:/10/25


# In[5]:


# Importing the Libraries


# In[3]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas

# In[46]:



import os


# In[47]:


os.getcwd()


# In[9]:


os.chdir('C:\\tanvi\\USER\\Desktop')


# In[10]:


data=pd.read_csv("heart.csv")


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


data.info()


# In[14]:


data.describe()


# In[15]:


data.shape


# In[16]:


data.size


# In[17]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment
# 

# In[18]:


# check Missing Value by record 

data.isna()


# In[19]:


data.isna().any()


# In[20]:


data.isna().sum()


# # Independent and Dependent Variables
# 

# In[22]:


x=data.drop("HeartDisease", axis=1)
y=data["HeartDisease"]


# # Splitting of DataSet into train and Test

# In[23]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# # Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression


# In[50]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in x_train.columns:
    if x_train[col].dtype == 'object':
        x_train[col] = le.fit_transform(x_train[col])
        x_test[col] = le.transform(x_test[col])


# In[28]:


x_train.dtypes


# In[29]:


from sklearn.linear_model import LogisticRegression


# In[31]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[32]:


y_pred1 = log.predict(x_test)


# In[33]:


from sklearn.metrics import accuracy_score 


# In[34]:


accuracy_score (y_test,y_pred1)


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[36]:


cm = confusion_matrix(y_test, y_pred1)


# In[37]:


labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)


# In[38]:


# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # KNN Classifier

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


# In[41]:


knn=KNeighborsClassifier()


# In[42]:


knn.fit(x_train, y_train)


# In[43]:


y_pred2=knn.predict(x_test)


# In[44]:


accuracy = accuracy_score(y_test, y_pred2)


# In[45]:


accuracy


# In[ ]:




