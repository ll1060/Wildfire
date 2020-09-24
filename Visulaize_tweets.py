import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/tweets_trunc_V2_no_trump.csv')

new_stopwords = ['http', 'co', 'RT','https'] + list(STOPWORDS)

wordcloud = WordCloud(background_color='white',stopwords=new_stopwords,width=800, height=400).generate(' '.join(df['text']))
plt.imshow(wordcloud)
plt.figure( figsize=(20,10) )
plt.show()
