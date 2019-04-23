#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier

# In[ ]:


# import pandas as pd
import numpy as np


# In[21]:


#import dataset using pandas
pulsar_data = pd.read_csv()
pulsar_data.head()


# In[22]:


#Preparing the data set
data_all = list(pulsar_data.shape)[0]
data_categories = list(pulsar_data['target_class'].value_counts())

print("The dataset has {} diagnosis, {} not star and {} star.".format(data_all, 
                                                                                 data_categories[0], 
                                                                                 data_categories[1]))


# In[23]:


X = pulsar_data.iloc[:, 1:-1].values
y = pulsar_data.iloc[:, 8].values


# In[24]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=600)
clf.fit(X_train,y_train)


# In[26]:


y_pred=clf.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix 
import numpy as np
print(np.mean(y_pred != y_test))


# In[20]:


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# In[ ]:




