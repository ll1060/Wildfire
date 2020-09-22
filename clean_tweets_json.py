import pandas as pd
# read the json file as csv and save it as df
# df = pd.read_json('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/tweets.json', lines = True)

df = pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/tweets_trunc_V2_no_trump.csv')

# pprint(data)
# print(df.head())
# df.to_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/tweets.csv')
# for col in df.columns:
#     print(col)
# df.drop('id_str',axis=1, inplace=True)
# df.drop('display_text_range',axis=1, inplace=True)
# df.drop('source',axis=1, inplace=True)
# df.drop('truncated',axis=1, inplace=True)
# df.drop('in_reply_to_status_id',axis=1, inplace=True)
# df.drop('in_reply_to_status_id_str',axis=1, inplace=True)
# df.drop('in_reply_to_user_id',axis=1, inplace=True)
# df.drop('in_reply_to_user_id_str',axis=1, inplace=True)
# df.drop('in_reply_to_screen_name',axis=1, inplace=True)
# df.drop('quoted_status_id',axis=1, inplace=True)
# df.drop('quoted_status_id_str',axis=1, inplace=True)
# df.drop('quoted_status',axis=1, inplace=True)
# df.drop('quoted_status_permalink',axis=1, inplace=True)
# df.drop('extended_entities',axis=1, inplace=True)
# df.drop('filter_level',axis=1, inplace=True)
# df.drop('entities',axis=1, inplace=True)
# df.drop('id',axis=1,inplace=True)
# df.drop('user',axis=1,inplace=True)
# df.drop('contributors',axis=1,inplace=True)
# for col in df.columns:
#     print(col)

# df.to_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/tweets_trunc_V1.csv')
# df.drop('retweeted_status',axis=1,inplace=True)
# df.drop('geo',axis=1,inplace=True)
# df.drop('coordinates',axis=1,inplace=True)
# df.drop('place',axis=1,inplace=True)
# df.drop('extended_tweet',axis=1,inplace=True)
# df.drop('quote_count',axis=1,inplace=True)
# df.drop('reply_count',axis=1,inplace=True)
# df.drop('retweet_count',axis=1,inplace=True)
# df.drop('favorite_count',axis=1,inplace=True)
# df.drop('favorited',axis=1,inplace=True)
# df.drop('retweeted',axis=1,inplace=True
# df.drop('timestamp_ms',axis=1, inplace=True)
# df.drop('possibly_sensitive',axis=1, inplace=True)
df.drop('Unnamed: 0',axis=1, inplace=True)
df.drop('Unnamed: 0.1.1.1.1.1',axis=1, inplace=True)

# delete rows that contain string 'ViolentTrump'
new_df = df[~df['text'].str.contains('ViolentTrump')]
# delete rows that where language of text isn't in English
new_df = df[df['lang'].str.contains('en')]
new_df.to_csv('/Users/lingfengcao/Leyao/ANLY501/Portfolio_Code_for_cleanData/tweets_trunc_V2_no_trump.csv')
