#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis of Random Forest Classifier
# 

# In[13]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:/10/25


# In[14]:


# Importing the Libraries


# In[15]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas 

# In[16]:


import os


# In[17]:


os.getcwd()


# In[20]:


os.chdir('C:\\tanvi\\USER\\Desktop')


# In[21]:


data=pd.read_csv("heart.csv")


# In[22]:


data.head()


# In[23]:


data.tail()


# In[24]:


data.info()


# In[25]:


data.describe()


# In[26]:


data.shape


# In[27]:


data.size


# In[28]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[29]:


# check Missing Value by record 

data.isna()


# In[30]:


data.isna().any()


# In[31]:


data.isna().sum()


# # Independent and Dependent Variables
# 

# In[33]:


x=data.drop("HeartDisease", axis=1)
y=data["HeartDisease"]


# # Splitting of DataSet into train and Test¶

# In[34]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# # Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in x_train.columns:
    if x_train[col].dtype == 'object':
        x_train[col] = le.fit_transform(x_train[col])
        x_test[col] = le.transform(x_test[col])


# In[47]:


x_train.dtypes


# In[48]:


from sklearn.metrics import accuracy_score 


# In[50]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[53]:


y_pred1 = log.predict(x_test)


# In[55]:


from sklearn.metrics import accuracy_score 


# In[56]:


accuracy_score (y_test,y_pred1)


# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[61]:


cm = confusion_matrix(y_test, y_pred1)


# In[62]:


labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)


# In[63]:


# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # KNN Classifier

# In[64]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


# In[65]:


knn=KNeighborsClassifier()


# In[66]:


knn.fit(x_train, y_train)


# In[67]:


y_pred2=knn.predict(x_test)


# In[68]:


accuracy = accuracy_score(y_test, y_pred2)


# In[69]:


accuracy


# # Support Vector Classifier / Machine (SVC/SVM)

# In[70]:


from sklearn import svm
svm=svm.SVC()
svm.fit(x_train, y_train)


# In[71]:


y_pred3=svm.predict(x_test)


# In[72]:


accuracy_score (y_test,y_pred3)


# # Decision Trees Algorithm

# In[73]:


from sklearn.tree import DecisionTreeClassifier


# In[74]:


dt=DecisionTreeClassifier()


# In[75]:


dt.fit(x_train, y_train)


# In[76]:


y_pred4=dt.predict(x_test)


# In[77]:


accuracy_score (y_test,y_pred4)


# In[ ]:





# # Random Forest Classifier

# In[78]:


from sklearn.ensemble import RandomForestClassifier


# In[79]:


rf=RandomForestClassifier()


# In[80]:


rf.fit(x_train, y_train)


# In[81]:



y_pred5=rf.predict(x_test)


# In[82]:


accuracy_score (y_test,y_pred5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




