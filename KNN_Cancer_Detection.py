#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np 


# In[21]:


#Step 1

#import dataset using pandas
wbcd = pd.read_csv()
wbcd.head()


# In[22]:


#Step 2

#Dropping the id feature 
wbcd.drop(columns = ['id'], inplace = True)
wbcd.head()

#Converting non-numeric data to numeric dataset
diag_map = {'M':1, 'B':0}
wbcd['diagnosis'] = wbcd['diagnosis'].map(diag_map)

#Preparing the data set
diagnosis_all = list(wbcd.shape)[0]
diagnosis_categories = list(wbcd['diagnosis'].value_counts())

print("The dataset has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all, 
                                                                                 diagnosis_categories[0], 
                                                                                 diagnosis_categories[1]))
wbcd.describe()


# In[23]:


#Normalizing numeric data
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min())) 
    dataNorm["diagnosis"]=dataset["diagnosis"]
    return dataNorm
wbcd_n=normalize(wbcd)
wbcd_n.describe()


# In[24]:


#Creating training and test datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

features_mean= list(wbcd_n.columns[1:31])
X = wbcd_n.loc[:,features_mean]
y = wbcd_n.loc[:, 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)


# In[25]:


#Step 3

#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=21)  
classifier.fit(X_train, y_train) 


# In[26]:


#Step 4

#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[19]:


#Step 5

#Improve Model Performance
#z-score transformed
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

#Training the model
X_train_z = scaler.transform(X_train)  
X_test_z = scaler.transform(X_test) 
classifier = KNeighborsClassifier(n_neighbors=21)  
classifier.fit(X_train_z, y_train) 

#Testing the model
y_pred_z = classifier.predict(X_test_z)  
print(np.mean(y_pred_z != y_test))
print(confusion_matrix(y_test, y_pred_z))  
print(classification_report(y_test, y_pred_z))  


# In[27]:


import matplotlib.pyplot as plt

error = []

# Calculating error for K values between 1 and 400
for i in range(1, 400):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 400), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 

plt.show()


# In[31]:


#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=55)  
classifier.fit(X_train, y_train) 


#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[32]:


#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=67)  
classifier.fit(X_train, y_train) 


#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[30]:


#Training Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=320)  
classifier.fit(X_train, y_train) 


#Testing the model
from sklearn.metrics import classification_report, confusion_matrix  
y_pred = classifier.predict(X_test)  
print(np.mean(y_pred != y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[ ]:




