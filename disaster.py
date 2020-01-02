import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk import TweetTokenizer, RegexpTokenizer
# import tensorflow as tf

df = pd.read_csv('datasets/train.csv')

# looking at distribution of targets
colors = ['#054e79', '#940b0b']
sns.countplot('target', data=df, palette=colors)
plt.title('Tweets about Disasters Distribution \n (0: Normal // 1: Disaster)')
# plt.show()
# ~ 40/60 real-to-normal split

# want y to be target, don't care about id
X = df.drop(['target', 'id'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# tokenize the text (separate into words)
tw_tk = TweetTokenizer(strip_handles=True, reduce_len=True)
# .loc is used to make sure we work on original dataframe, ignore warning message
X_train.loc[:, 'text'] = X_train['text'].apply(tw_tk.tokenize)
print(X_train['text'].head())

# clean up the text by removing punctuation and stop-words, and converting all words to lower case




