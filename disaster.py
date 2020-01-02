import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

