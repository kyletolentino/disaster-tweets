import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from text_processing import preprocess_text

# my process: load dataframe, preprocess dataframe, then convert?

df = pd.read_csv('../datasets/train.csv')

train_df, test_df = train_test_split(df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)


def df_to_ds(dataframe):
    target = dataframe.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((dataframe.values, target))
    return dataset

train_df['text'] = train_df['text'].apply(preprocess_text)
print(train_df['text'])


