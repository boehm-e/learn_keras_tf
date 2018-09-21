print("BEGIN ...")
import time

dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]


import time
import tensorflow as tf
print("DEBUG 0 ...")
from keras.datasets import cifar10
print("DEBUG 1 ...")
from keras.preprocessing.image import ImageDataGenerator
print("DEBUG 2 ...")
from keras.models import Sequential
print("DEBUG 3 ...")
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
print("DEBUG 4 ...")
from keras.callbacks import TensorBoard
print("DEBUG 5 ...")
import pickle

NAME = "Cats-vs-Dogs-cnn-64x2_{}".format(int(time.time()))

print("LOADING DATAS...")

sess = tf.Session(config=tf.ConfigProto())

X = pickle.load(open("../02/X.pickle", "rb"))
y = pickle.load(open("../02/y.pickle", "rb"))

# resize images to 0-1
X = X/255.0

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            print("DEBUG LOOP ...")
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            NAME = "{}___{}_conv-{}_nodes-{}_dense".format(int(time.time()), conv_layer, layer_size, dense_layer)
            print(NAME)

            model = Sequential()

            model.add( Conv2D(64, (3,3), input_shape = X[0].shape) )
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1): # -1 bcs we already the one beside
                model.add( Conv2D(64, (3,3)) )
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("relu"))

            model.compile(
                loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])
