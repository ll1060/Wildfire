#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from textblob import TextBlob
import numpy as np
import sklearn



import matplotlib.pyplot as plt
from sklearn.feature_extraction import text

df = pd.read_csv('labeled_forestfires_v3.csv')
# df.head()
df.drop(['Unnamed: 0','X','Y'],inplace=True,axis=1)
df.drop(['Unnamed: 0.1'],inplace=True,axis=1)
df.drop(['area'],inplace=True,axis=1)
df.head()


# In[5]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['area_label'] = label_encoder.fit_transform(df['area_label'])

df.head()


# In[ ]:





# In[6]:


from sklearn.model_selection import train_test_split
import random as rd
rd.seed(77)
TrainDF, TestDF = train_test_split(df, test_size=0.2)
train_data = TrainDF.drop(["area_label"], axis=1)
train_label = TrainDF["area_label"]
test_data = TestDF.drop(["area_label"], axis=1)
test_true_label = TestDF["area_label"]
# train_data.head()
# train_label[1:10]


# In[7]:



from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix



# In[8]:


TestDF.head()


# In[9]:





myNB = MultinomialNB()

myNB.fit(train_data, train_label)
NB_pred = myNB.predict(test_data)


# In[10]:


from sklearn import metrics


auc_score = metrics.accuracy_score(test_true_label, NB_pred)
print(auc_score)


# In[11]:


labels = ['small','medium','large']
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


# In[12]:




############# SVM ##############
from sklearn.svm import LinearSVC
SVM_model=LinearSVC(C=10)
SVM_model.fit(train_data, train_label)


SVM_pred=SVM_model.predict(test_data)




auc_score_SVM = metrics.accuracy_score(label_encoder.inverse_transform(test_true_label), label_encoder.inverse_transform(SVM_pred))
print(auc_score_SVM)


# In[13]:




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


# In[14]:


from sklearn import svm
rbf = svm.SVC(kernel='rbf', gamma=1, C=10, decision_function_shape='ovo').fit(train_data,train_label)
pred_rbf = rbf.predict(test_data)
poly = svm.SVC(kernel='poly', degree=4, C=10, decision_function_shape='ovo').fit(train_data,train_label)
pred_poly = poly.predict(test_data)
sig = svm.SVC(kernel='sigmoid', C=10, decision_function_shape='ovo').fit(train_data,train_label)
pred_sig = sig.predict(test_data)


auc_score_SVM_rbf = metrics.accuracy_score(test_true_label, pred_rbf)
auc_score_SVM_poly = metrics.accuracy_score(test_true_label, pred_poly)
auc_score_SVM_sig = metrics.accuracy_score(test_true_label, pred_sig)

labels = ['small','large']
cm_rbf = confusion_matrix(label_encoder.inverse_transform(test_true_label), label_encoder.inverse_transform(pred_rbf),labels)
cm_poly = confusion_matrix(label_encoder.inverse_transform(test_true_label), label_encoder.inverse_transform(pred_poly),labels)
cm_sig = confusion_matrix(label_encoder.inverse_transform(test_true_label), label_encoder.inverse_transform(pred_sig),labels)

print(cm_rbf)
print(cm_poly)
print(cm_sig)


# In[15]:



print(auc_score_SVM_rbf)
print(auc_score_SVM_poly)
print(auc_score_SVM_sig)


# In[19]:


# Plot Decision Region using mlxtend's awesome plotting function
#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(X=np.array(train_data['DC']), 
        #              y=np.array(train_label),
      #                clf=SVM_model, 
      #                legend=2)

# Update plot object with X/Y axis labels and Figure Title
#plt.xlabel(X.columns[0], size=14)
#plt.ylabel(X.columns[1], size=14)
#plt.title('SVM Decision Region Boundary', size=16)

#import os
#os.getcwd()


# In[69]:





# In[70]:





# In[74]:





# In[75]:





# In[ ]:




