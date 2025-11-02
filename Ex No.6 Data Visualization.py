#!/usr/bin/env python
# coding: utf-8

# In[4]:


#name: Tanvi Shelke
#roll no: 58
#subject:Dss
#date:6/10/25


# In[2]:


#aim: data visualition using matplotlib


# In[5]:


#importing the basic library
import numpy as np
from matplotlib import pyplot as plt


# In[6]:


x=np.arange(1,11)


# In[7]:


x


# In[9]:


y=2*x


# In[10]:


y


# #Line chart

# In[20]:


plt.plot(x,y)
plt.title("Line chart")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


# In[38]:


plt.bar(x,y, color=["#AEC6CF","#FFDAB9"])
plt.title("Bar chart")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


# In[48]:


a=(1,2,4,5,6,7,8,3)
b=(34,56,87,92,23,45,61,53)
plt.scatter(a,b,color="green",s=120)
plt.title("Scatter chart")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


# In[75]:


labels=["banana","cherries","dates"]
sizes = [25, 30, 20]
colors=["blue","green","pink"]
plt.pie(sizes, labels=labels, colors=colors,startangle=60)
plt.title("pie chart")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


# In[63]:




labels = ["Apples", "Bananas", "Cherries", "Dates"]
sizes = [25, 30, 20, 19]  # Percentage
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]  # pastel colors
explode = (0.1, 0, 0, 0)  # "explode" first slice

# Create pie chart
plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90)

plt.title("Fruit Distribution")
plt.show()


# In[65]:



data = [12, 15, 12, 18, 20, 25, 20, 19, 18, 15, 12, 22, 25, 20, 12]

# Create histogram
plt.hist(data, bins=5, color="#90EE90", edgecolor="black")  # bins = number of intervals

# Add labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram - Matplotlib")

plt.show()


# In[ ]:




