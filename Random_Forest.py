#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


#import dataset using pandas
glass_data = pd.read_csv("C:/Users/chink/Documents/Northeastern University/Predictive_Analytics_ALY6020/Week_4/Assignment/glass.csv")
glass_data.head()


# In[4]:


glass_data.Type.replace([1], [0], inplace=True)
glass_data.Type.replace([2], [1], inplace=True)
glass_data.Type.replace([3], [2], inplace=True)
glass_data.Type.replace([5], [3], inplace=True)
glass_data.Type.replace([6], [4], inplace=True)
glass_data.Type.replace([7], [5], inplace=True)


# In[5]:


#Preparing the data set
data_all = list(glass_data.shape)[0]
class_categories = list(glass_data['Type'].value_counts())

print("The dataset has {} types, {} as 1, {} as 0, {} as 5, {} as 2 and {} as 3 and {} as 4.".format(data_all, 
                                                            class_categories[0], 
                                                            class_categories[1],
                                                            class_categories[2],
                                                            class_categories[3],
                                                            class_categories[4],
                                                            class_categories[5]))


# In[6]:


feature_cols = ['RI', 'Na', 'Mg', 'Al','Si','K','Ca','Ba','Fe']

X = glass_data[feature_cols]
y = glass_data["Type"]


# In[7]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 2)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=900)
clf.fit(X_train,y_train)


# In[9]:


y_pred=clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix 
import numpy as np
print(np.mean(y_pred != y_test))


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix 
import numpy as np
print(np.mean(y_pred != y_test))


# In[11]:


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# In[12]:


clf.feature_importances_


# In[13]:


feature_imp = pd.Series(clf.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[15]:


X_train = X_train.drop(columns = ['Fe'])
X_test = X_test.drop(columns = ['Fe'])


# In[16]:


import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# In[293]:


param = {
    'max_depth': 5,  # the maximum depth of each tree
    'eta': 0.20,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 6}  # the number of classes that exist in this datset
num_round = 7  # the number of training iterations


# In[294]:


bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)


# In[295]:


import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])


# In[296]:


print(np.mean(best_preds != y_test))


# In[297]:


print(confusion_matrix(y_test, best_preds))  
print(classification_report(y_test, best_preds)) 


# In[ ]:




