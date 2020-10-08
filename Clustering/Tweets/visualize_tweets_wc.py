
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS


df = pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/HW MOD03/tweets_python/tweets_labeled_clean.csv')

# pos_tweet = df['clean_tweet'].where(df['sentiment_label'] == 'positive')
pos_tweet = df.loc[df.sentiment_label =='positive']['clean_tweet']
ntra_tweet =  df.loc[df.sentiment_label =='neutral']['clean_tweet']
neg_tweet =  df.loc[df.sentiment_label =='negative']['clean_tweet']
# print(pos_tweet)
# new_df=pd.concat([df, pos_tweet],axis=1)
# print(new_df.head())
wordcloud1 = WordCloud(background_color='white',width=800, height=400).generate(' '.join(pos_tweet))
wordcloud2 = WordCloud(background_color='white',width=800, height=400).generate(' '.join(ntra_tweet))
wordcloud3 = WordCloud(background_color='white',width=800, height=400).generate(' '.join(neg_tweet))
# wordcloud = WordCloud(background_color='white',width=800, height=400).generate(' '.join(new_df['clean_tweet']))
plt.imshow(wordcloud1)
plt.imshow(wordcloud2)
plt.imshow(wordcloud3)
# plt.figure( figsize=(20,10) )
plt.show()
