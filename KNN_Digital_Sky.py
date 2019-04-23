#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 


# In[3]:


#Step 1

#import dataset using pandas
sky_data = pd.read_csv()
sky_data.head()


# In[4]:


#Step 2

#Dropping the id feature
sky_data.drop(columns = ['objid'], inplace = True)
sky_data.head()

#Converting non-numeric data to numeric dataset
diag_map = {'STAR':1, 'GALAXY':2, 'QSO':3}
sky_data['class'] = sky_data['class'].map(diag_map)

#Preparing the data set
class_all = list(sky_data.shape)[0]
class_categories = list(sky_data['class'].value_counts())

print("The dataset has {} classes, {} stars, {} galaxies and {} quasars.".format(class_all, 
                                                                                 class_categories[0], 
                                                                                 class_categories[1],
                                                                                 class_categories[2]))
sky_data.describe()


# In[5]:


#Creating training and test datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

y = sky_data["class"].values
X = sky_data.drop(["class"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)


# In[6]:


#Step 3

#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train) 


# In[7]:


#Step 4

#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[8]:


#Step 5

#Improve Model Performance
#z-score transformed
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

#Training the model
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train) 

#Testing the model
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[21]:


import matplotlib.pyplot as plt

error = []

# Calculating error for K values between 1 and 300
for i in range(1, 300):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 300), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 

plt.show()


# In[22]:


#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=11)  
classifier.fit(X_train, y_train) 

#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[13]:


#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=50)  
classifier.fit(X_train, y_train)

#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[14]:


#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=100)  
classifier.fit(X_train, y_train) 


#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[ ]:




