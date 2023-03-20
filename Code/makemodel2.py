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
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

# default mode / type of vulnerability
mode = "sql"

# get the vulnerability from the command line argument
if (len(sys.argv) > 1):
    mode = sys.argv[1]


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

with open('data/' + mode + '_dataset_keystrain', 'rb') as infile:
    keystrain = json.load(infile)
with open('data/' + mode + '_dataset_keystest', 'rb') as infile:
    keystest = json.load(infile)
with open('data/' + mode + '_dataset_keysfinaltest', 'rb') as infile:
    keysfinaltest = json.load(infile)

with open('data/' + mode + 'allblocks', 'rb') as infile:
    allblocks = json.load(infile)


X = []
y = []
TrainX = []
TrainY = []
ValidateX = []
ValidateY = []
FinaltestX = []
FinaltestY = []



print("Creating  dataset... (" + mode + ")")
for k in tqdm(allblocks):
  block = allblocks[k]
  code = block[0]
  token = myutils.getTokens(code) #get all single tokens from the snippet of code
  vectorlist = []
  for t in tqdm(token, leave=False): #convert all tokens into their word2vec vector representation
    if t in word_vectors.key_to_index and t != " ":
      vector = word_vectors[t]
      vectorlist.append(vector.tolist())
  X.append(vectorlist) #append the list of vectors to the X (independent variable)
  y.append(block[1]) #append the label to the Y (dependent variable)


print("Creating training dataset... (" + mode + ")")
for k in tqdm(keystrain):
  block = allblocks[k]
  code = block[0]
  token = myutils.getTokens(code) #get all single tokens from the snippet of code
  vectorlist = []
  for t in tqdm(token, leave=False): #convert all tokens into their word2vec vector representation
    if t in word_vectors.key_to_index and t != " ":
      vector = word_vectors[t]
      vectorlist.append(vector.tolist())
  TrainX.append(vectorlist) #append the list of vectors to the X (independent variable)
  TrainY.append(block[1]) #append the label to the Y (dependent variable)

print("Creating validation dataset...")
for k in tqdm(keystest):
  block = allblocks[k]
  code = block[0]
  token = myutils.getTokens(code) #get all single tokens from the snippet of code
  vectorlist = []
  for t in tqdm(token, leave=False): #convert all tokens into their word2vec vector representation
    if t in word_vectors.key_to_index and t != " ":
      vector = word_vectors[t]
      vectorlist.append(vector.tolist())
  ValidateX.append(vectorlist) #append the list of vectors to the X (independent variable)
  ValidateY.append(block[1]) #append the label to the Y (dependent variable)

print("Creating finaltest dataset...")
for k in tqdm(keysfinaltest):
  block = allblocks[k]
  code = block[0]
  token = myutils.getTokens(code) #get all single tokens from the snippet of code
  vectorlist = []
  for t in tqdm(token, leave=False): #convert all tokens into their word2vec vector representation
    if t in word_vectors.key_to_index and t != " ":
      vector = word_vectors[t]
      vectorlist.append(vector.tolist())
  FinaltestX.append(vectorlist) #append the list of vectors to the X (independent variable)
  FinaltestY.append(block[1]) #append the label to the Y (dependent variable)

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now() # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)


# saving samples

with open('data/plain_' + mode + '_dataset-train-X_'+w2v + "__" + mode2, 'wb') as fp:
 pickle.dump(TrainX, fp)
with open('data/plain_' + mode + '_dataset-train-Y_'+w2v + "__" + mode2, 'wb') as fp:
 pickle.dump(TrainY, fp)
with open('data/plain_' + mode + '_dataset-validate-X_'+w2v + "__" + mode2, 'wb') as fp:
 pickle.dump(ValidateX, fp)
with open('data/plain_' + mode + '_dataset-validate-Y_'+w2v + "__" + mode2, 'wb') as fp:
 pickle.dump(ValidateY, fp)
with open('data/' + mode + '_dataset_finaltest_X', 'wb') as fp:
  pickle.dump(FinaltestX, fp)
with open('data/' + mode + '_dataset_finaltest_Y', 'wb') as fp:
  pickle.dump(FinaltestY, fp)
print("saved finaltest.")

# Prepare the data for the LSTM model

X = numpy.array(X)
numpy.save(f"data/Dataset_X_{mode}", X)
y = numpy.array(X)
numpy.save(f"data/Dataset_y_{mode}", y)

X_train = numpy.array(TrainX)
y_train = numpy.array(TrainY)
X_test = numpy.array(ValidateX)
y_test = numpy.array(ValidateY)
X_finaltest = numpy.array(FinaltestX)
y_finaltest = numpy.array(FinaltestY)

numpy.save(f'data/X_train_{mode}_{w2v}_{mode2}.npy', X_train)
numpy.save(f'data/y_train_{mode}_{w2v}_{mode2}.npy', y_train)
numpy.save(f'data/X_test{mode}_{w2v}_{mode2}.npy', X_test)
numpy.save(f'data/y_test{mode}_{w2v}_{mode2}.npy', y_test)
numpy.save(f'data/X_finaltest{mode}_{w2v}_{mode2}.npy', X_finaltest)
numpy.save(f'data/y_finaltest{mode}_{w2v}_{mode2}.npy', y_finaltest)

# in the original collection of data, the 0 and 1 were used the other way round, so now they are switched so that "1" means vulnerable and "0" means clean.

for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = 1
    else:
        y_train[i] = 0

for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = 1
    else:
        y_test[i] = 0

for i in range(len(y_finaltest)):
    if y_finaltest[i] == 0:
        y_finaltest[i] = 1
    else:
        y_finaltest[i] = 0

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")


