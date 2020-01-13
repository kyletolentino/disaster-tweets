import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')

y = train['target'].values
tweet_train = train['text'].values
tweet_test = test['text'].values

tk = Tokenizer()
tk.fit_on_texts(list(tweet_train))
tokenized_train = tk.texts_to_sequences(tweet_train)
tokenized_test = tk.texts_to_sequences(tweet_test)
vocab_size = len(tk.word_index) + 1
# vocab size is 19424

total = [len(i) for i in tweet_train]
# plt.figure(1)
# plt.hist(total, bins=np.arange(0, 410, 10))
# plt.show()
# 150 looks like a good maximum (most were around 100-150)

max_len = 150
X_train = tf.keras.preprocessing.sequence.pad_sequences(tokenized_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(tokenized_test, maxlen=max_len)

# i = Input(shape=(max_len, ))
embed_size = 128

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embed_size, input_shape=(max_len, )),
    layers.LSTM(16, return_sequences=True, name='lstm_layer'),
    layers.GlobalMaxPool1D(),
    layers.Dropout(0.1),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid', name='out_layer')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
model.summary()

batch_size = 64
epoch_size = 10

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y, epochs=epoch_size, batch_size=batch_size,
                    validation_split=0.25, callbacks=[callback], shuffle=True)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
