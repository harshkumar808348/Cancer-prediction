#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns


# In[2]:


data = pd.read_csv("data.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


#data cleaning


# In[7]:


#remove the empty field 


# In[8]:


sns.heatmap(data.isnull())


# In[9]:


data.drop(["Unnamed: 32" , "id"] ,axis=1 , inplace=True)


# In[10]:


#agar naya data bayaege kuch change karne ke bad to inplca true to likhna parta hai 


# In[11]:


data.head()


# In[12]:


data.diagnosis = [1 if value =="M" else 0 for value in data.diagnosis]


# In[13]:


#bascially we change if m 1 then 1 else 0 


# In[14]:


data.head()


# In[15]:


data.diagnosis


# In[16]:


data["diagnosis"] = data["diagnosis"].astype("category" , copy = False)
data["diagnosis"].value_counts().plot(kind = "bar")


# In[17]:


#data divide into target varibale and predictable variable 


# In[18]:


data.diagnosis


# In[19]:


y = data["diagnosis"] # target variable 
X = data.drop(["diagnosis"] , axis=1)


# In[20]:


X


# In[21]:


# i want to normalise the data due to points in this 17.99 and 10.57 due to decimal points in this 


# In[22]:


from sklearn.preprocessing import  StandardScaler


#create sclaer object
scaler = StandardScaler()

#fit the scaler to the data and transform the data 
X_scaled = scaler.fit_transform(X)


# In[23]:


X


# In[24]:


X_scaled


# In[25]:


#split the data 


# In[26]:


from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(X_scaled,y, test_size = 0.30, random_state =42 )


# In[27]:


#train the model


# In[28]:


from sklearn.linear_model import LogisticRegression

#create the data 
lr   = LogisticRegression()

#train the model on training data 
lr.fit(X_train , y_train)

#predict the target variable on tets data 
y_pred = lr.predict(X_test)


# In[29]:


y_pred


# In[30]:


y_test


# In[31]:


#evaluation of the model 


# In[32]:


from sklearn.metrics import accuracy_score


# In[33]:


accuracy =  accuracy_score(y_test , y_pred)


# In[34]:


print(f"Accuracy: {accuracy: 2f}")


# In[35]:


from sklearn.metrics import classification_report 
print(classification_report(y_test , y_pred))


# In[36]:


print("Columns in training data (X):", X.columns)


# In[37]:


new_data = pd.DataFrame({
    'radius_mean': [12.5],
    'texture_mean': [14.8],
    'perimeter_mean': [85.0],
    'area_mean': [525.0],
    'smoothness_mean': [0.09],
    'compactness_mean': [0.12],
    'concavity_mean': [0.1],
    'concave points_mean': [0.07],
    'symmetry_mean': [0.18],
    'fractal_dimension_mean': [0.062],
    'radius_se': [0.2],
    'texture_se': [0.5],
    'perimeter_se': [1.2],
    'area_se': [15.0],
    'smoothness_se': [0.006],
    'compactness_se': [0.02],
    'concavity_se': [0.03],
    'concave points_se': [0.01],
    'symmetry_se': [0.02],
    'fractal_dimension_se': [0.003],
    'radius_worst': [13.5],
    'texture_worst': [16.5],
    'perimeter_worst': [92.5],
    'area_worst': [635.0],
    'smoothness_worst': [0.1],
    'compactness_worst': [0.15],
    'concavity_worst': [0.2],
    'concave points_worst': [0.1],
    'symmetry_worst': [0.25],
    'fractal_dimension_worst': [0.08]
})


# In[38]:


new_data_scaled = scaler.transform(new_data)


# In[39]:


new_prediction = lr.predict(new_data_scaled)


# In[40]:


if new_prediction[0] == 1:
    print("The prediction is: Malignant (M)")
else:
    print("The prediction is: Benign (B)")


# In[47]:


import shutil
import os
from IPython.display import FileLink


# In[48]:


# Replace with your folder path
folder_path = '/Machine Learning Model/cancer prediction by logistic regression'

# Define the path to your Downloads folder
downloads_path = os.path.expanduser('~/Downloads')  # For Mac/Linux
# downloads_path = r'C:\Users\<Your-Username>\Downloads'  # For Windows

# Define the path where you want to save the ZIP file in the Downloads folder
zip_file_path = os.path.join(downloads_path, 'your_folder.zip')

# Create the ZIP archive
shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', folder_path)

# Provide a link to download the ZIP file
FileLink(zip_file_path)


# In[ ]:




