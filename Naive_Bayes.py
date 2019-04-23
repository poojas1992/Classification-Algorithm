#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Step 1

#import dataset using pandas
sms_raw = pd.read_csv()
sms_raw.head()


# In[3]:


#Preparing the data set
sms_all = list(sms_raw.shape)[0]
type_categories = list(sms_raw['type'].value_counts())

print("The dataset has {} types, {} ham and {} spam.".format(sms_all, 
                                                            type_categories[0], 
                                                            type_categories[1]))


# In[32]:


#Word Cloud
ham_words = ''
spam_words = ''
spam = sms_raw[sms_raw.type == "spam"]
ham = sms_raw[sms_raw.type == "ham"]


# In[36]:


import nltk
from nltk.corpus import stopwords
nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize

for val in spam.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    for words in tokens:
        spam_words = spam_words + words + ' '
        
for val in ham.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '


# In[41]:


from wordcloud import WordCloud

spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)


# In[42]:


#Spam Word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[43]:


#Ham word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[5]:


X = sms_raw.iloc[:, 1].values  
y = sms_raw.iloc[:, 0].values  


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X)
X_counts.shape


# In[14]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_tfidf.shape


# In[16]:


X_tfidf


# In[21]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25)


# In[22]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)


# In[23]:


import numpy as np
y_pred = clf.predict(X_test)
np.mean(y_pred == y_test)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix  


# In[25]:


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[27]:


x=["Winner! You have won 10,000$. Call +18038989067 to claim the prize"]
c = count_vect.transform(x)
c_tfidf = tfidf_transformer.transform(c)
clf.predict(c_tfidf)


# In[ ]:




