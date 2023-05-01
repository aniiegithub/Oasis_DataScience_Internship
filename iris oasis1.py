#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[10]:


df = pd.read_csv(r"Iris.csv")


# In[11]:


print (df)


# In[12]:


df.head


# In[13]:


df.tail


# In[14]:


df['PetalWidthCm']


# In[15]:


projected_columns = ['PetalWidthCm','Species']
df[projected_columns]


# In[16]:


df['Species'].value_counts()


# In[17]:


input_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df_input = df[input_features]
df_input


# In[18]:


output_features = ['Species']
df_output = df[output_features]
df_output


# In[19]:


a, b, c = [2, 3, 1]
     


# In[20]:


df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(df_input, df_output, test_size = 0.2)
     


# In[21]:


df_output_test


# In[39]:


import sklearn.model_selection
import numpy as np
model = KNeighborsClassifier()


# In[26]:


df_output_test


# In[29]:


data = {
    'Id': [23, 24, 12, 56, 87],
    'Gender': ['M', 'F', 'F', 'M', 'F']
}

df = pd.DataFrame(data)

gender_map = {
    'M':  0,
    'F': 1
}

df['Gender'] = df['Gender'].map(gender_map)

df.head()


# In[32]:


data = {
    'Age': [2, np.nan, 5, 1, 4],
    'Weight': [np.nan, 3, 6, np.nan, 1]
}

df = pd.DataFrame(data)

df.head()


# In[33]:


df.isnull().sum()


# In[ ]:




