import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def prepare_for_cam(gray):
    IMG_SIZE = 28
    new_array = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    cv2.imshow('frame',new_array)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE)


model = tf.keras.models.load_model("./RNN_mnist.model")
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = prepare_for_cam(gray) / 255.0
    predict = model.predict([img])
    print(np.argmax(predict[0]))
    print("====")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
