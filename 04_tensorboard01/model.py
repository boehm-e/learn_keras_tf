import time
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle

NAME = "Cats-vs-Dogs-cnn-64x2_{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

sess = tf.Session(config=tf.ConfigProto())

X = pickle.load(open("../02/X.pickle", "rb"))
y = pickle.load(open("../02/y.pickle", "rb"))

# resize images to 0-1
X = X/255.0
# print(X[0].shape)
# > (50, 50, 1) | 50x50 grayscale images

model = Sequential()

model.add( Conv2D(64, (3,3), input_shape = X[0].shape) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(64, (3,3)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])
model.save('catsVsDog.model')
