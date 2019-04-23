#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd


# In[48]:


#import dataset using pandas
mushroom_data = pd.read_csv()
mushroom_data.head()


# In[49]:


#Preparing the data set
data_all = list(mushroom_data.shape)[0]
class_categories = list(mushroom_data['class'].value_counts())

print("The dataset has {} types, {} Edible and {} Poisonous.".format(data_all, 
                                                            class_categories[0], 
                                                            class_categories[1]))


# In[50]:


mushroom_data = mushroom_data.apply(lambda x: pd.factorize(x)[0])


# In[51]:


#split dataset in features and target variable
X = mushroom_data.iloc[:, 1:23]
y = mushroom_data.iloc[:,0]


# In[52]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[53]:


# Create Decision Tree classifer object and training it
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)
dtree.fit(X_train, y_train)


# In[54]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report 
y_pred = dtree.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))   
print(classification_report(y_test, y_pred))  


# In[56]:


import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
feature_names = X.columns

dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=['0','1'])
graph = graphviz.Source(dot_data)  
graph.render("mushroom_tree",view="True")


# In[ ]:




