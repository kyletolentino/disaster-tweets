import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')

# looking at distribution of targets
# colors = ['#054e79', '#940b0b']
# sns.countplot('target', data=train, palette=colors)
# plt.title('Tweets about Disasters Distribution \n (0: Normal // 1: Disaster)')
# plt.figure()
# plt.show()
# ~ 40/60 real-to-normal split

train_df, val_df = train_test_split(train, test_size=0.2, random_state=1)

# tokenize and update vocabulary based on training text
tk = Tokenizer()
train_text = train_df['text'].to_numpy()
tk.fit_on_texts(train_text)

vocab_size = len(tk.word_index) + 1
# print(vocab_size)


# after tokenizing text, want to convert into sequences (i.e. mapping text to a seq of integers)
def text_to_seq(text):
    seq = tk.texts_to_sequences(text)
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post')
    return seq


train_seq = text_to_seq(train_text)
train_target = train_df['target'].to_numpy()

val_text = val_df['text'].to_numpy()
val_seq = text_to_seq(val_text)
val_target = val_df['target'].to_numpy()

test_text = test['text'].to_numpy()
test_seq = text_to_seq(test_text)

val_ds = tf.data.Dataset.from_tensor_slices((val_seq, val_target))

# creating datasets
BUFFER_SIZE = 6090
BATCH_SIZE = 32
# batch size of 32 outperformed other powers of 2

train_ds = tf.data.Dataset.from_tensor_slices((train_seq, train_target))
# print(tf.data.experimental.cardinality(train_ds))

train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# prefetching is used to decouple the time between when data is produced and consumed
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# build a sequential model for RNN
model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 512),
    layers.Bidirectional(tf.keras.layers.LSTM(32)),
    layers.Dropout(0.1),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[callback])


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


test['target'] = model.predict(test_seq)
print(test.head())