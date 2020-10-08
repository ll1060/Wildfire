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

## perform sentiment analysis on original data and create label for the sentiment result
df =  pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/HW MOD03/tweets_python/tweets_trunc_V2_no_trump.csv')

## perform sentiment analysis on each tweet
def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None

df['sentiment_polarity'] = df['text'].apply(sentiment_calc)

## create 3 labels for tweets sentiment
def label_sentiment (row):
   if row['sentiment_polarity'] > 0 :
      return 'positive'
   if row['sentiment_polarity'] == 0:
      return 'neutral'
   if row['sentiment_polarity'] < 0 :
      return 'negative'

df['sentiment_label'] = df.apply (lambda row: label_sentiment(row), axis=1)
## save this labeled dataframe
# df.to_csv('/Users/lingfengcao/Leyao/ANLY501/HW MOD03/tweets_python/tweets_labeled.csv')

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet

def remove_emoji(tweet):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',tweet)

my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_emoji(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet
df['clean_tweet'] = df.text.apply(clean_tweet)




# add some extra stopwords
extra_stopwords = ['oh','p','pa','of','o','h','f','#','a','b','c','d','e','g','i','j','k','l',
                  'm','n','q','r','s','t','u','v','w','x','y','z','’r','’m','’t', '’v', '“', '“i',
                  '±', '\u200d♀️', '\u200d🧑\u200d\u200d','“the','…', 'ⓟⓞⓖⓞⓗⓤⓑ', '▫️', '▫️follow',
                   '▫️like', '▶️', '♀️', '⚠️⚠️⚠️', '⚠️⚠️⚠️donat', '⛰️', '⛷️', '✅', '✅over', '✔️',
                   '✨', '❌', '❤️', '️', '𝗖𝗮𝗺𝗽', '𝗙𝗼𝗿𝗲𝘀𝘁', '𝗜𝗻𝘁𝗿𝗼𝗱𝘂𝗰𝗶𝗻𝗴', '𝗞𝗶', '𝗦𝗰𝗵𝗼𝗼𝗹', '𝗪𝗮', '𝗬', '𝗮𝘁', '𝘈',
                  '🆕', '\U0001f7e0⚪', '🤔', '🤥', '🤫rioter', '\U0001f971', '🥳', '🥺','#…', '#𝔊𝕯𝔏𝔈']

# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+', max_features=2000,
                            stop_words=text.ENGLISH_STOP_WORDS.union(extra_stopwords))
vectorizer_tfidf=TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(extra_stopwords),
                            max_features=2000)

# get a document term matrix
# and a a matrix of TF-IDF features
dtm = vectorizer.fit_transform(df['clean_tweet']).toarray()
dtm_tf=vectorizer_tfidf.fit_transform(df['clean_tweet']).toarray()
#  get a list of words in the files
vocab = vectorizer.get_feature_names()

## calculate distance matrix using Euclidean distance metric
## round to 2 decimal places
dist = euclidean_distances(dtm)
np.round(dist,2)

## calculate the cosine distance
## round to 2 decimal places
cosdist = 1 - cosine_similarity(dtm)
np.round(cosdist,2)


## Use pandas to create data frames
df_count=pd.DataFrame(dtm,columns=vocab)
df_tf=pd.DataFrame(dtm_tf,columns=vocab)

kmeans_count = sklearn.cluster.KMeans(n_clusters=5)
kmeans_count.fit(df_count)

labels = kmeans_count.labels_
prediction_kmeans = kmeans_count.predict(df_count)

# --------------- plot 3 words in 3d --------------- #
## plot 3 words of choice in 3d
# print(df_count.columns)
# get locations of these following words
# df_count.columns.get_loc("acre")
# df_count.columns.get_loc("wildfire")
# df_count.columns.get_loc("ca")
# x = df_count["acre"]## col 91
# y = df_count["wildfire"]## col 1914
# z = df_count["ca"]## col 323
#
#
# fig1 = plt.figure(figsize=(12, 12))
# ax1 = Axes3D(fig1, rect=[0, 0, .90, 1], elev=48, azim=134)
#
# ax1.scatter(x, y, z, cmap="RdYlGn", edgecolor='k', s=200, c=prediction_kmeans)
# ax1.w_xaxis.set_ticklabels([])
# ax1.w_yaxis.set_ticklabels([])
# ax1.w_zaxis.set_ticklabels([])
#
# ax1.set_xlabel('acre', fontsize=25)
# ax1.set_ylabel('wildfire', fontsize=25)
# ax1.set_zlabel('ca', fontsize=25)
# plt.show()
# plt.close()


centers = kmeans_count.cluster_centers_

C1 = centers[0, (91, 1914, 323)]

C2 = centers[1, (91, 1914, 323)]

xs = C1[0], C2[0]
ys = C1[1], C2[1]
zs = C1[2], C2[2]
fig1 = plt.figure(figsize=(12, 12))
ax1 = Axes3D(fig1, rect=[0, 0, .90, 1], elev=48, azim=134)
ax1.scatter(xs, ys, zs, c='black', s=2000, alpha=0.2)
plt.show()


## perform hierarchical clustering
# hierarchical plot
Z = linkage(cosdist, 'ward')
plt.title("Ward")
dendrogram(Z, labels=labels)
plt.show()
