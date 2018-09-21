# usefull for analyzing timeseries data (when order of the data is important : nlp, speech)

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM # CuDNN LSTM
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM

mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0


# build model

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(X_train[0].shape), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))


optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-5) # decay: smaller the learning rate with time
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))
model.save('RNN_mnist.model')
