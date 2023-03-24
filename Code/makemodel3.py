import tensorflow
from imblearn.over_sampling import SMOTE

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
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from keras.utils.data_utils import pad_sequences


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

w2v_model = Word2Vec.load(w2vmodel)
word_vectors = w2v_model.wv

# X = numpy.load(f"data/Dataset_X_{mode}")
# y = numpy.load(f"data/Dataset_y_{mode}")
print("Loading X_train, y_train. ")
X_train = numpy.load(f'data/X_train_{mode}_{w2v}_{mode2}.npy', allow_pickle=True)
y_train = numpy.load(f'data/y_train_{mode}_{w2v}_{mode2}.npy', allow_pickle=True)

# print(X_train.shape)
# print(type(X_train))
# # print(X_train[0])
#
# X_train = pad_sequences(X_train, maxlen=fulllength)
# X_train = tensorflow.cast(X_train, tensorflow.float32)
#
# print(X_train.shape)
# print(X_train[0])
#
# exit()

print(type(y_train))
print(y_train.shape)
print(numpy.unique(y_train, return_counts=True))
print(y_train[0])
print("Loading X_test, y_test. ")
X_test = numpy.load(f'data/X_test{mode}_{w2v}_{mode2}.npy', allow_pickle=True)
y_test = numpy.load(f'data/y_test{mode}_{w2v}_{mode2}.npy', allow_pickle=True)
print("Loading X_finaltest, y_finaltest. ")
X_finaltest = numpy.load(f'data/X_finaltest{mode}_{w2v}_{mode2}.npy', allow_pickle=True)
y_finaltest = numpy.load(f'data/y_finaltest{mode}_{w2v}_{mode2}.npy', allow_pickle=True)


now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")




csum = 0
for a in y_train:
    csum = csum + a
print("percentage of vulnerable samples: " + str(int((csum / len(X_train)) * 10000) / 100) + "%")

testvul = 0
for y in y_test:
    if y == 1:
        testvul = testvul + 1
print("absolute amount of vulnerable samples in test set: " + str(testvul))

max_length = fulllength

# hyperparameters for the LSTM model

dropout = 0.2
neurons = 10
optimizer = "adam"
epochs = 10
batchsize = 128

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Starting LSTM: ", nowformat)

print("Dropout: " + str(dropout))
print("Neurons: " + str(neurons))
print("Optimizer: " + optimizer)
print("Epochs: " + str(epochs))
print("Batch Size: " + str(batchsize))
print("max length: " + str(max_length))

# padding sequences on the same length
now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Start padding squenceses: ", nowformat)

X_train = pad_sequences(X_train, maxlen=max_length)
X_train = tensorflow.cast(X_train, tensorflow.float32)

# print(X_train[0])
# print(X_train.shape)
#
#
#
# smote = SMOTE(random_state=2021)
# X_train, y_train = smote.fit_resample(X_train, y_train)

X_test = pad_sequences(X_test, maxlen=max_length)
X_finaltest = pad_sequences(X_finaltest, maxlen=max_length)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("End padding squenceses: ", nowformat)

# creating the model
now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Start compile LSTM: ", nowformat)

model = Sequential()
model.add(LSTM(neurons, dropout=dropout, recurrent_dropout=dropout))  # around 50 seems good
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=myutils.f1_loss, optimizer='adam', metrics=[myutils.f1])

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Compiled LSTM: ", nowformat)

# account with class_weights for the class-imbalanced nature of the underlying data
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=numpy.unique(y_train), y=y_train)

# training the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, verbose=1,)
                    # class_weight=class_weights)  # epochs more are good, batch_size more is good

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("saving LSTM model " + mode + ". ", nowformat)
model.save('model/LSTM_model_' + mode + '.h5')  # creates a HDF5 file 'my_model.h5'
print("\n\n")
# validate data on train and test set

for dataset in ["train", "test", "finaltest"]:
    print("Now predicting on " + dataset + " set (" + str(dropout) + " dropout)")

    if dataset == "train":
        # yhat_classes = numpy.argmax(model.predict(X_train, verbose=1), axis=1)
        yhat_classes = (model.predict(X_train, verbose=1) > 0.5).astype("int32")

        print("yhat_classes")
        print(type(yhat_classes))
        print(yhat_classes.shape)
        print(numpy.unique(yhat_classes, return_counts=True))
        print(yhat_classes[0])
        accuracy = accuracy_score(y_train, yhat_classes)
        precision = precision_score(y_train, yhat_classes)
        recall = recall_score(y_train, yhat_classes)
        F1Score = f1_score(y_train, yhat_classes)

    if dataset == "test":
        # yhat_classes = numpy.argmax(model.predict(X_test, verbose=1), axis=1)
        yhat_classes = (model.predict(X_test, verbose=1) > 0.5).astype("int32")
        print("yhat_classes")
        print(type(yhat_classes))
        print(yhat_classes.shape)
        print(numpy.unique(yhat_classes, return_counts=True))
        print(yhat_classes[0])
        accuracy = accuracy_score(y_test, yhat_classes)
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        F1Score = f1_score(y_test, yhat_classes)


    if dataset == "finaltest":
        # yhat_classes = numpy.argmax(model.predict(X_finaltest, verbose=1), axis=1)
        yhat_classes = (model.predict(X_finaltest, verbose=1) > 0.5).astype("int32")
        accuracy = accuracy_score(y_finaltest, yhat_classes)
        precision = precision_score(y_finaltest, yhat_classes)
        recall = recall_score(y_finaltest, yhat_classes)
        F1Score = f1_score(y_finaltest, yhat_classes)

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print('F1 score: %f' % F1Score)
    print("\n")

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("saving LSTM model " + mode + ". ", nowformat)
model.save('model/LSTM_model_' + mode + '.h5')  # creates a HDF5 file 'my_model.h5'
print(" \n\n")