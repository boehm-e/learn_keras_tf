import pandas as pd
import time
from sklearn import preprocessing
from collections import deque
import collections
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization # BatchNormalization: normalize data from previous layer to next
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


# predict LTC-USD close (price) 3min in the future with 60min previous data of LTC-BTC-ETH and BCH (bcs each row is a minute)
FUTURE_PERIOD_PREDICT = 3
SEQ_LEN = 240
CURRENCY_TO_PREDICT = "BTC-USD"

EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{CURRENCY_TO_PREDICT}-{SEQ_LEN}-SEQ--{FUTURE_PERIOD_PREDICT}-PRED--{int(time.time())}"

# to classify the dataset
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future', 1)
    # scale data
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change() # pandas : return percentage change from the previous row
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values) # standardize the datas : (from 0 to 1 )
    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)# create an array with max row = 60 => [ [0,1,..,60], [61,62,...,120] ]
    for i in df.values: # df.values is the dataframe without index: everything but time
        prev_days.append([n for n in i[:-1]]) # append everything but target (last index)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]]) # from 60 past data try to predict i[-1] (target)
    random.shuffle(sequential_data)

    buys = []
    sells = []

    # find the number of sells, and the number of buys (when to buy : 0 and when not to: 1)
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    # keep se same amount of buys and sells:
    buys = buys[:lower]
    sells = sells[:lower]

    # now sequential_data will have the same amount of buys and sells ( if it has like 70 / 30 ratio, it will be a bad model as it would try to classify the 70 bcause it is more likely to encounter it)
    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    # find the lower values

    # assert len(buys) == len(sells)
    print("OK")
    return np.array(X), np.array(y)



main_df = pd.DataFrame()

currencies = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
# MERGE DFs
for currency in currencies:
    dataset = f"crypto_data/{currency}.csv"
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{currency}_close", "volume": f"{currency}_volume"}, inplace=True)
    df.set_index("time", inplace=True)

    # only keep close and volume (close is the price)
    df = df[[f"{currency}_close", f"{currency}_volume"]]

    # now merge DFs
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.dropna(inplace = True)


# create new column "future" and set it to the value in 3 min
main_df["future"] = main_df[f"{CURRENCY_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
# print(main_df[[f"{CURRENCY_TO_PREDICT}_close", "future"]].head())

# create new column "target" : 1 if the close increase in the x future minut, 0 if not
main_df["target"] = list(map(classify, main_df[f"{CURRENCY_TO_PREDICT}_close"], main_df["future"]))
# print(main_df[[f"{CURRENCY_TO_PREDICT}_close", "future", "target"]].head(20))


# RNN are for time series datas (like crypto datas) so we just cannot shuffle the data
# to get testing data, a good way to do is to take the x% last values as training data

times = sorted(main_df.index.values) # get the sorted timestamps (they should already be)
last_5_pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5_pct)]
main_df = main_df[(main_df.index < last_5_pct)]


train_X, train_y = preprocess_df(main_df)
validation_X, validation_y = preprocess_df(validation_main_df)

print(f"train data : {len(train_X)} validation: {len(validation_X)}")
print(f"Dont buy : {collections.Counter(train_y)[0]} buy: {collections.Counter(train_y)[1]}")
print(f"VALIDATION Dont buy : {collections.Counter(validation_y)[0]} buy: {collections.Counter(validation_y)[1]}")


model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_X.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_X.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_X.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer = opt,
    metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones


history = model.fit(
    train_X, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_X, validation_y),
    callbacks=[tensorboard, checkpoint])
