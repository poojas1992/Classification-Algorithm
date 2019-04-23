#!/usr/bin/env python
# coding: utf-8

# In[247]:


import pandas as pd


# In[248]:


#import dataset using pandas
loan_data = pd.read_csv()
loan_data.head()


# In[249]:


#Preparing the data set
data_all = list(loan_data.shape)[0]
class_categories = list(loan_data['default'].value_counts())

print("The dataset has {} types, {} No and {} Yes.".format(data_all, 
                                                            class_categories[0], 
                                                            class_categories[1]))


# In[250]:


loan_data['checking_balance'],_ = pd.factorize(loan_data['checking_balance'])  
loan_data['credit_history'],_ = pd.factorize(loan_data['credit_history']) 
loan_data['purpose'],_ = pd.factorize(loan_data['purpose']) 
loan_data['savings_balance'],_ = pd.factorize(loan_data['savings_balance']) 
loan_data['employment_length'],_ = pd.factorize(loan_data['employment_length']) 
loan_data['personal_status'],_ = pd.factorize(loan_data['personal_status']) 
loan_data['other_debtors'],_ = pd.factorize(loan_data['other_debtors']) 
loan_data['property'],_ = pd.factorize(loan_data['property']) 
loan_data['installment_plan'],_ = pd.factorize(loan_data['installment_plan']) 
loan_data['housing'],_ = pd.factorize(loan_data['housing']) 
loan_data['telephone'],_ = pd.factorize(loan_data['telephone']) 
loan_data['foreign_worker'],_ = pd.factorize(loan_data['foreign_worker'])
loan_data['job'],_ = pd.factorize(loan_data['job'])


# In[237]:


loan_data.info()


# In[306]:


#split dataset in features and target variable
feature_cols = ['checking_balance', 'months_loan_duration', 'credit_history', 'purpose','amount','savings_balance',
                'employment_length', 'installment_rate', 'personal_status', 'other_debtors', 'residence_history', 
                'property', 'age', 'installment_plan', 'housing', 'existing_credits','dependents', 'telephone',
                'foreign_worker', 'job']
X = loan_n[feature_cols] 
y = loan_n.default 


# In[307]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[308]:


#Check if the training data is split well
y_train.value_counts()


# In[309]:


# Create Decision Tree classifer object and training it
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtree_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
dtree_entropy.fit(X_train, y_train)


# In[310]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report 
y_pred = dtree_entropy.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[261]:


dtree_entropy.feature_importances_
feature_imp = pd.Series(dtree_entropy.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp


# In[315]:


feature_cols = ['checking_balance', 'months_loan_duration', 'credit_history', 'purpose','amount','savings_balance',
                'residence_history', 'property', 'installment_plan', 'existing_credits', 'job', 'employment_length',
               'other_debtors']
X = loan_n[feature_cols] 


# In[316]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[322]:


# Create Decision Tree classifer object and training it
#from sklearn.tree import DecisionTreeClassifier
dtree_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=17,  random_state=1)
dtree_entropy.fit(X_train, y_train)


# In[323]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report 
y_pred = dtree_entropy.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[325]:


import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
feature_names = X.columns

dot_data = tree.export_graphviz(dtree_entropy, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=['1','2'])
graph = graphviz.Source(dot_data)  
graph.render("dtree_entropy",view = True)


# In[ ]:




