#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis Support Vector Machine

# In[2]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:/10/25


# In[3]:


# Importing the Libraries


# In[4]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas

# In[6]:


import os


# In[7]:


os.getcwd()


# In[8]:


os.chdir('C:\\tanvi\\USER\\Desktop')


# In[9]:


data=pd.read_csv("heart.csv")


# In[10]:


data.head()


# In[11]:


data.tail()


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


data.shape


# In[15]:


data.size


# In[16]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment
# 

# In[17]:


# check Missing Value by record 

data.isna()


# In[18]:


data.isna().any()


# In[19]:


data.isna().sum()


# # Independent and Dependent Variables

# In[20]:


x=data.drop("HeartDisease", axis=1)
y=data["HeartDisease"]


# # Splitting of DataSet into train and Test

# In[21]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# # Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression


# In[25]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in x_train.columns:
    if x_train[col].dtype == 'object':
        x_train[col] = le.fit_transform(x_train[col])
        x_test[col] = le.transform(x_test[col])


# In[26]:


x_train.dtypes


# In[49]:


from sklearn.linear_model import LogisticRegression


# In[50]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[29]:


y_pred1 = log.predict(x_test)


# In[30]:


from sklearn.metrics import accuracy_score 


# In[31]:


accuracy_score (y_test,y_pred1)


# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[33]:


cm = confusion_matrix(y_test, y_pred1)
                      


# In[35]:


labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)


# In[36]:


# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # KNN Classifier

# In[37]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


# In[38]:


knn=KNeighborsClassifier()


# In[39]:


knn.fit(x_train, y_train)


# In[40]:


y_pred2=knn.predict(x_test)


# In[41]:


accuracy = accuracy_score(y_test, y_pred2)


# In[42]:


accuracy


# # Support Vector Classifier / Machine (SVC/SVM)

# In[48]:


from sklearn import svm
svm=svm.SVC()
svm.fit(x_train, y_train)


# In[46]:


y_pred3=svm.predict(x_test)


# In[47]:


accuracy_score (y_test,y_pred3)


# In[ ]:




