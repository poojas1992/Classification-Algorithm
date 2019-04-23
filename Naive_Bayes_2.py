#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[52]:


#import dataset using pandas
twitter_data = pd.read_csv()
twitter_data.head()


# In[53]:


#Preparing the data set
twitter_all = list(twitter_data.shape)[0]
class_categories = list(twitter_data['type'].value_counts())

print("The dataset has {} types, {} Negatives and {} Positives.".format(twitter_all, 
                                                            class_categories[0], 
                                                            class_categories[1]))


# In[54]:


#Word Cloud
Pos_words = ''
Neg_words = ''
Pos = twitter_data[twitter_data.type == "Pos"]
Neg = twitter_data[twitter_data.type == "Neg"]


# In[55]:


import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize

for val in Neg.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    for words in tokens:
        Neg_words = Neg_words + words + ' '
        
for val in Pos.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        Pos_words = Pos_words + words + ' '


# In[56]:


from wordcloud import WordCloud

neg_wordcloud = WordCloud(width=600, height=400).generate(Neg_words)
pos_wordcloud = WordCloud(width=600, height=400).generate(Pos_words)


# In[57]:


#Spam Word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(neg_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[58]:


#Ham word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(pos_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[59]:


X = twitter_data.iloc[:, 1].values  
y = twitter_data.iloc[:, 0].values  


# In[60]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X)
X_counts.shape


# In[61]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_tfidf.shape


# In[62]:


X_tfidf


# In[63]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20)


# In[64]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)


# In[65]:


import numpy as np
y_pred = clf.predict(X_test)
np.mean(y_pred == y_test)


# In[66]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[67]:


x=["The @apple music isnt the greatest software in the current market."]
c = count_vect.transform(x)
c_tfidf = tfidf_transformer.transform(c)
clf.predict(c_tfidf)


# In[ ]:




