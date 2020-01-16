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

# total = [len(i) for i in tweet_train]
# plt.figure(1)
# plt.hist(total, bins=np.arange(0, 410, 10))
# plt.show()
# 150 looks like a good maximum (most were around 100-150)

# (7613, 150) shape for X_train
max_len = 150
X_train = tf.keras.preprocessing.sequence.pad_sequences(tokenized_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(tokenized_test, maxlen=max_len)

embed_size = 256
epoch_size = 50

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embed_size, input_shape=(max_len, )),
    layers.GlobalMaxPool1D(),
    layers.Dropout(0.1),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(momentum=0.9, nesterov=True, lr=0.005, decay=1e-6),
              metrics=['accuracy'])
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y, epochs=epoch_size,
                    validation_split=0.2, callbacks=[callback], shuffle=True)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

