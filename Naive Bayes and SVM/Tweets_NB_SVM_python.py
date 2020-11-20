#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from textblob import TextBlob
import numpy as np
import sklearn
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction import text

df = pd.read_csv('tweets_labeled_clean.csv')


df = df[['sentiment_label','clean_tweet']]


# In[2]:


extra_stopwords = ['oh','p','pa','of','o','h','f','#','a','b','c','d','e','g','i','j','k','l',
                  'm','n','q','r','s','t','u','v','w','x','y','z','’r','’m','’t', '’v', '“', '“i',
                  '±', '\u200d♀️', '\u200d🧑\u200d\u200d','“the','…', 'ⓟⓞⓖⓞⓗⓤⓑ', '▫️', '▫️follow',
                   '▫️like', '▶️', '♀️', '⚠️⚠️⚠️', '⚠️⚠️⚠️donat', '⛰️', '⛷️', '✅', '✅over', '✔️',
                   '✨', '❌', '❤️', '️', '𝗖𝗮𝗺𝗽', '𝗙𝗼𝗿𝗲𝘀𝘁', '𝗜𝗻𝘁𝗿𝗼𝗱𝘂𝗰𝗶𝗻𝗴', '𝗞𝗶', '𝗦𝗰𝗵𝗼𝗼𝗹', '𝗪𝗮', '𝗬', '𝗮𝘁', '𝘈',
                  '🆕', '\U0001f7e0⚪', '🤔', '🤥', '🤫rioter', '\U0001f971', '🥳', '🥺','#…', '#𝔊𝕯𝔏𝔈','#']
# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+', max_features=500,
                            stop_words=text.ENGLISH_STOP_WORDS.union(extra_stopwords))
vectorizer_tfidf=TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(extra_stopwords),
                            max_features=500)

# get a document term matrix
# and a a matrix of TF-IDF features
dtm = vectorizer.fit_transform(df['clean_tweet'])
dtm_tf=vectorizer_tfidf.fit_transform(df['clean_tweet']).toarray()
#  get a list of words in the files
vocab = vectorizer.get_feature_names()
vocab_tf = vectorizer_tfidf.get_feature_names()


# In[3]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
sentiment_encoded = label_encoder.fit_transform(df['sentiment_label'])
builderS = pd.DataFrame(dtm.toarray(),columns=vocab)
builderS['Label']=sentiment_encoded
# builderS.head()
builderTS=pd.DataFrame(dtm_tf,columns=vocab_tf)
builderTS['Label']=df['sentiment_label']
builderS.head()


# In[4]:


FinalDF=pd.DataFrame()
FinalDF= FinalDF.append(builderS)
FinalDF_TFIDF=pd.DataFrame()
FinalDF_TFIDF= FinalDF_TFIDF.append(builderTS)


# In[5]:


from sklearn.model_selection import train_test_split
import random as rd
rd.seed(77)
TrainDF, TestDF = train_test_split(FinalDF, test_size=0.3)
train_data = TrainDF.drop(["Label"], axis=1)
train_label = TrainDF["Label"]
test_data = TestDF.drop(["Label"], axis=1)
test_true_label = TestDF["Label"]
# train_data.head()
# train_label[1:10]


# In[6]:



from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix



# In[7]:


TestDF.head()


# In[8]:





myNB = MultinomialNB()

myNB.fit(train_data, train_label)
NB_pred = myNB.predict(test_data)


# In[19]:


from sklearn import metrics


auc_score = metrics.accuracy_score(test_true_label, NB_pred)
print(auc_score)


# In[18]:


labels = ['negative', 'neutral', 'positive']
NB_conmat= confusion_matrix(label_encoder.inverse_transform(test_true_label), label_encoder.inverse_transform(NB_pred),labels)
print(NB_conmat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(NB_conmat)
plt.title('Confusion matrix of Multinomial Naive Bayes')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[23]:




############# SVM ##############
from sklearn.svm import LinearSVC
SVM_model=LinearSVC(C=10)
SVM_model.fit(train_data, train_label)


SVM_pred=SVM_model.predict(test_data)

SVM_conmat = confusion_matrix(test_true_label, SVM_pred)
print(SVM_conmat)


auc_score_SVM = metrics.accuracy_score(test_true_label, SVM_pred)
print(auc_score_SVM)


# In[29]:



labels = ['negative', 'neutral', 'positive']
SVM_conmat= confusion_matrix(label_encoder.inverse_transform(test_true_label), label_encoder.inverse_transform(SVM_pred),labels)
print(SVM_conmat)
fig_2 = plt.figure()
ax = fig_2.add_subplot(111)
cax_SVM = ax.matshow(SVM_conmat)
plt.title('Confusion matrix of SVM')
fig.colorbar(cax_SVM)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[47]:





# In[20]:





# In[ ]:




