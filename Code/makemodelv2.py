import tensorflow
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import myutils
import sys
import os.path
import json
from datetime import datetime
import random
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
# from keras import backend as K
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from tqdm import tqdm
from keras.utils.data_utils import pad_sequences
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from keras.layers import CuDNNLSTM as LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set GPU number to 0
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# K.set_session(session)

import pandas as pd

# default mode / type of vulnerability
mode = "sql"

# get the vulnerability from the command line argument
if (len(sys.argv) > 1):
    mode = sys.argv[1]

progress = 0
count = 0

### paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step lenght n in the description
fulllength = 200  # context length m in the description

mode2 = str(step) + "_" + str(fulllength)

### hyperparameters for the w2v model
mincount = 10  # minimum times a word has to appear in the corpus to be in the word2vec model
iterationen = 50  # training iterations for the word2vec model
s = 200  # dimensions of the word2vec model
w = "withString"  # word2vec model is not replacing strings but keeping them

# get word2vec model
w2v = "word2vec_" + w + str(mincount) + "-" + str(iterationen) + "-" + str(s)
w2vmodel = "w2v/" + w2v + ".model"

# load word2vec model
if not (os.path.isfile(w2vmodel)):
    print("word2vec model is still being created...")
    sys.exit()

w2v_model = KeyedVectors.load(w2vmodel)
word_vectors = w2v_model.wv

# load data
with open(f'data/plain_{mode}', 'r') as infile:
    data = json.load(infile)

nowformat = datetime.now().strftime("%H:%M")
print(f"finished loading. {nowformat}")

allblocks = []

for row in tqdm(data, total=len(data)):
    for col, col_data in data[row].items():
        if "files" not in col_data:
            continue
        for f, file_data in col_data["files"].items():
            if "source" not in file_data:
                continue
            sourcecode = file_data["source"]
            allbadparts = [bad for change in file_data["changes"] for bad in change.get("badparts", [])]
            count += len(allbadparts)
            positions = myutils.findpositions(allbadparts, sourcecode)
            blocks = myutils.getblocks(sourcecode, positions, step, fulllength)
            allblocks += blocks if blocks else []

keys = np.arange(len(allblocks))
np.random.shuffle(keys)

cutoff = round(0.7 * len(keys))
cutoff2 = round(0.85 * len(keys))
keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]


def create_sequences(keys, blocks):
    X, y = [], []
    for k in tqdm(keys):
        block = blocks[k]
        code = block[0]
        token = myutils.getTokens(code)
        vectorlist = [word_vectors[t].tolist() for t in token if t in word_vectors.key_to_index and t != " "]
        X.append(vectorlist)
        y.append(block[1])

        # Pad sequences
        # with tf.device('/GPU:0'):
    X = pad_sequences(X, maxlen=fulllength, padding='pre', truncating='pre', dtype='float32')

    # print(X.shape)
    # # Reshape to 2D
    # X = X.reshape(X.shape[0], X.shape[1] * X.shape[1])
    # print(X.shape)
    #
    # # Apply standardization
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # print(X.shape)
    #
    # # Apply SMOTE
    # smote = SMOTE()
    # X, y = smote.fit_resample(X, y)
    # print(X.shape)
    #
    # # Reshape back to original 3D shape
    # X = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))
    # print(X.shape)
    # # Print mean and standard deviation
    # print("Mean:", np.mean(X))
    # print("Standard deviation:", np.std(X))

    y = np.where(np.array(y) == 0, 1, 0)
    return X, y


def create_sequences_train(keys, blocks):
    X, y = [], []
    for k in tqdm(keys):
        block = blocks[k]
        code = block[0]
        token = myutils.getTokens(code)
        vectorlist = [word_vectors[t].tolist() for t in token if t in word_vectors.key_to_index and t != " "]
        X.append(vectorlist)
        y.append(block[1])

    # Convert X to a Python list

    # Pad sequences
    X = pad_sequences(X, maxlen=fulllength, padding='pre', truncating='pre', dtype='float32')

    print(X.shape)
    # Reshape to 2D
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[1])
    print(X.shape)

    # Apply standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X.shape)

    # Apply SMOTE
    # smote = SMOTE()
    # X, y = smote.fit_resample(X, y)
    # print(X.shape)

    # Reshape back to original 3D shape
    X = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))
    print(X.shape)
    # Print mean and standard deviation
    print("Mean:", np.mean(X))
    print("Standard deviation:", np.std(X))

    y = np.where(np.array(y) == 0, 1, 0)
    return X, y


X_train, y_train = create_sequences_train(keystrain, allblocks)
X_test, y_test = create_sequences(keystest, allblocks)
X_finaltest, y_finaltest = create_sequences(keysfinaltest, allblocks)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")

import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.layers import Input, Dense, LSTM
import matplotlib.pyplot as plt
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
# from keras.preprocessing.sequence import pad_sequences
from imblearn.keras import BalancedBatchGenerator

csum = np.sum(y_train)
print(f"percentage of vulnerable samples: {int((csum / len(X_train)) * 10000) / 100}%")

testvul = np.sum(y_test)
print(f"absolute amount of vulnerable samples in test set: {testvul}")

max_length = fulllength

# hyperparameters for the LSTM model
dropout = 0.2
neurons = 10
optimizer = "adam"
epochs = 1
batchsize = 32

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Starting LSTM: ", nowformat)

print("Dropout: " + str(dropout))
print("Neurons: " + str(neurons))
print("Optimizer: " + optimizer)
print("Epochs: " + str(epochs))
print("Batch Size: " + str(batchsize))
print("max length: " + str(max_length))

# #padding sequences on the same length
# X_train = pad_sequences(X_train, maxlen=max_length)
# X_test = pad_sequences(X_test, maxlen=max_length)
# X_finaltest = pad_sequences(X_finaltest, maxlen=max_length)

# with tf.device('/GPU:0'):
training_generator = TimeseriesGenerator(X_train, y_train, batch_size=32, length=max_length)

# creating the model
# with tf.device('/GPU:0'):
model = Sequential()
model.add(LSTM(neurons, dropout=dropout, recurrent_dropout=dropout))  # around 50 seems good
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=tf.metrics.binary_crossentropy, optimizer='adam', metrics=[myutils.f1])

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Compiled LSTM: ", nowformat)

# account with class_weights for the class-imbalanced nature of the underlying data
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=numpy.unique(y_train), y=y_train)
print(type(class_weights))
class_weights = dict(enumerate(class_weights))
print(type(class_weights))
print(class_weights)

# training the model
# with tf.device('/GPU:0'):
model.fit(X_train, y_train, epochs=epochs, verbose=1, batch_size=batchsize, workers=8, use_multiprocessing=True,
          class_weight=class_weights)
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, class_weight=class_weights) #epochs more are good, batch_size more is good

# from sklearn.naive_bayes import GaussianNB
#
# model = GaussianNB()
#
# model.fit(X_train, y_train)

# validate data on train and test set


for dataset in ["train", "test", "finaltest"]:
    print("Now predicting on " + dataset + " set (" + str(dropout) + " dropout)")

    if dataset == "train":
        # yhat_classes = numpy.argmax(model.predict(X_train, verbose=1), axis=1)
        yhat_classes = (model.predict(X_train, verbose=1) > 0.5).astype("int32")

        accuracy = accuracy_score(y_train, yhat_classes)
        precision = precision_score(y_train, yhat_classes)
        recall = recall_score(y_train, yhat_classes)
        F1Score = f1_score(y_train, yhat_classes)
        res = tf.math.confusion_matrix(y_train, yhat_classes)

    if dataset == "test":
        # yhat_classes = numpy.argmax(model.predict(X_test, verbose=1), axis=1)
        yhat_classes = (model.predict(X_test, verbose=1) > 0.5).astype("int32")

        accuracy = accuracy_score(y_test, yhat_classes)
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        F1Score = f1_score(y_test, yhat_classes)
        res = tf.math.confusion_matrix(y_test, yhat_classes)

    if dataset == "finaltest":
        # yhat_classes = numpy.argmax(model.predict(X_finaltest, verbose=1), axis=1)
        yhat_classes = (model.predict(X_finaltest, verbose=1) > 0.5).astype("int32")
        accuracy = accuracy_score(y_finaltest, yhat_classes)
        precision = precision_score(y_finaltest, yhat_classes)
        recall = recall_score(y_finaltest, yhat_classes)
        F1Score = f1_score(y_finaltest, yhat_classes)
        res = tf.math.confusion_matrix(y_finaltest, yhat_classes)

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print('F1 score: %f' % F1Score)
    print("Confusion Matrix: " + str(res))
    print("\n")

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("saving LSTM model " + mode + ". ", nowformat)
model.save('model/LSTM_model_' + mode + '.h5')  # creates a HDF5 file 'my_model.h5'
print(" \n\n")
