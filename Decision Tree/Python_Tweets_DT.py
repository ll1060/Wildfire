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


# In[35]:


extra_stopwords = ['oh','p','pa','of','o','h','f','#','a','b','c','d','e','g','i','j','k','l',
                  'm','n','q','r','s','t','u','v','w','x','y','z','’r','’m','’t', '’v', '“', '“i',
                  '±', '\u200d♀️', '\u200d🧑\u200d\u200d','“the','…', 'ⓟⓞⓖⓞⓗⓤⓑ', '▫️', '▫️follow',
                   '▫️like', '▶️', '♀️', '⚠️⚠️⚠️', '⚠️⚠️⚠️donat', '⛰️', '⛷️', '✅', '✅over', '✔️',
                   '✨', '❌', '❤️', '️', '𝗖𝗮𝗺𝗽', '𝗙𝗼𝗿𝗲𝘀𝘁', '𝗜𝗻𝘁𝗿𝗼𝗱𝘂𝗰𝗶𝗻𝗴', '𝗞𝗶', '𝗦𝗰𝗵𝗼𝗼𝗹', '𝗪𝗮', '𝗬', '𝗮𝘁', '𝘈',
                  '🆕', '\U0001f7e0⚪', '🤔', '🤥', '🤫rioter', '\U0001f971', '🥳', '🥺','#…', '#𝔊𝕯𝔏𝔈','#']
# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+', max_features=20,
                            stop_words=text.ENGLISH_STOP_WORDS.union(extra_stopwords))
vectorizer_tfidf=TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(extra_stopwords),
                            max_features=20)

# get a document term matrix
# and a a matrix of TF-IDF features
dtm = vectorizer.fit_transform(df['clean_tweet']).toarray()
dtm_tf=vectorizer_tfidf.fit_transform(df['clean_tweet']).toarray()
#  get a list of words in the files
vocab = vectorizer.get_feature_names()
vocab_tf = vectorizer_tfidf.get_feature_names()


# In[36]:


builderS = pd.DataFrame(dtm,columns=vocab)
builderS['Label']=df['sentiment_label']
# builderS.head()
builderTS=pd.DataFrame(dtm_tf,columns=vocab_tf)
builderTS['Label']=df['sentiment_label']
builderTS.head()


# In[37]:


FinalDF=pd.DataFrame()
FinalDF= FinalDF.append(builderS)
FinalDF_TFIDF=pd.DataFrame()
FinalDF_TFIDF= FinalDF_TFIDF.append(builderTS)


# In[38]:


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


# In[39]:



from sklearn import tree
from sklearn.metrics import confusion_matrix
import graphviz 
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[40]:





myDT =DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

myDT.fit(train_data, train_label)
tree.plot_tree(myDT)


# In[41]:


from sklearn import metrics
#prediction
DT_pred=myDT.predict(test_data)
print(DT_pred)
## Show the confusion matrix
bn_matrix = confusion_matrix(test_true_label, DT_pred)
print(bn_matrix)
auc_score = metrics.accuracy_score(test_true_label, DT_pred)
print(auc_score)


# In[42]:


dot_data = tree.export_graphviz(myDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=eval(str('train_data'+".columns")),  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
graphviz.Source(dot_data).view()
tempname=str("graph_50fetures")
# graph.render(tempname) 
# from dtreeviz.trees import dtreeviz # remember to load the package

# viz = dtreeviz(myDT, X, y,
#                 target_name="target",
#                 feature_names=iris.feature_names,
#                 class_names=list(iris.target_names))

# viz


# In[43]:


feature_names = []
for col in train_data:
    feature_names.append(col)
# print(len(feature_names))
FeatureImp=myDT.feature_importances_   
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(train_data.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])



# In[44]:


# Random Forest
# import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

RF = RandomForestClassifier()
RF.fit(train_data, train_label)
RF_pred=RF.predict(test_data)



# In[45]:



# Random Forest confusion matrix and acurracy
bn_matrix_RF_text = confusion_matrix(test_true_label, RF_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF_text)
auc_score_RF = metrics.accuracy_score(test_true_label, RF_pred)
print(auc_score_RF)


# In[47]:


figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(RF.estimators_[0],
               feature_names = feature_names, 
               #class_names=Targets,
               filled = True)

##save it
figT.savefig('RF_Tree_Text_2')  ## creates png


# In[20]:


feature_names[0:10]


# In[ ]:




