# IGNORE THIS UNTIL I FIND SOLUTION FOR ENCODING/DECODING

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

df = pd.read_csv('../datasets/train.csv')

# looking at distribution of targets
colors = ['#054e79', '#940b0b']
sns.countplot('target', data=df, palette=colors)
plt.title('Tweets about Disasters Distribution \n (0: Normal // 1: Disaster)')
# plt.show()
# ~ 40/60 real-to-normal split

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.text, target.values))

# 7613 rows, let's do a 80/20 train-test split
# 7613 * .8 = 6090 (roughly)
train_dataset = dataset.take(6090)
test_dataset = dataset.skip(6090)

