# gonna try using tf hub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('../datasets/train.csv')

# looking at distribution of targets
colors = ['#054e79', '#940b0b']
sns.countplot('target', data=df, palette=colors)
plt.title('Tweets about Disasters Distribution \n (0: Normal // 1: Disaster)')
# plt.show()
# ~ 40/60 real-to-normal split

target = df.pop('target')
# print(df.head())

# want to do a 80/20 train-test split
train_size = int(len(df) * .8)

train_df = df[:train_size]
train_target = target[:train_size]

test_df = df[train_size:]
test_target = df[train_size:]

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.KerasLayer(module_url)

labels = ['Normal', 'Disaster']

encoder = MultiLabelBinarizer()
encoder.fit(labels)
train_encoded = encoder.transform(test_target)
test_encoded = encoder.transform(test_target)
num_classes = len(encoder.classes_)

print(encoder.classes_)
