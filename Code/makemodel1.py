import myutils
import sys
import os.path
import json
from datetime import datetime
import random
import pickle
from tqdm import tqdm
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

# load data
with open('data/plain_' + mode, 'r') as infile:
    data = json.load(infile)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

allblocks = []

for row in tqdm(data, total=len(data)):
    progress = progress + 1

    for col in data[row]:

        if "files" in data[row][col]:
            #  if len(data[r][c]["files"]) > restriction[3]:
            # too many files
            #    continue

            for f in data[row][col]["files"]:

                #      if len(data[r][c]["files"][f]["changes"]) >= restriction[2]:
                # too many changes in a single file
                #       continue

                if not "source" in data[row][col]["files"][f]:
                    # no sourcecode
                    continue

                if "source" in data[row][col]["files"][f]:
                    sourcecode = data[row][col]["files"][f]["source"]
                    #     if len(sourcecode) > restriction[0]:
                    # sourcecode is too long
                    #       continue

                    allbadparts = []

                    for change in data[row][col]["files"][f]["changes"]:

                        # get the modified or removed parts from each change that happened in the commit
                        badparts = change["badparts"]
                        count = count + len(badparts)

                        #     if len(badparts) > restriction[1]:
                        # too many modifications in one change
                        #       break

                        for bad in badparts:
                            # check if they can be found within the file
                            pos = myutils.findposition(bad, sourcecode)
                            if not -1 in pos:
                                allbadparts.append(bad)

                    #   if (len(allbadparts) > restriction[2]):
                    # too many bad positions in the file
                    #     break

                    if (len(allbadparts) > 0):
                        #   if len(allbadparts) < restriction[2]:
                        # find the positions of all modified parts
                        positions = myutils.findpositions(allbadparts, sourcecode)

                        # get the file split up in samples
                        blocks = myutils.getblocks(sourcecode, positions, step, fulllength)

                        for b in blocks:
                            # each is a tuple of code and label
                            allblocks.append(b)

keys = []

# randomize the sample and split into train, validate and final test set
for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))  # 70% for the training set
cutoff2 = round(0.85 * len(keys))  # 15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]

print("cutoff " + str(cutoff))
print("cutoff2 " + str(cutoff2))

with open('data/' + mode + '_dataset_keystrain', 'wb') as fp:
    pickle.dump(keystrain, fp)
with open('data/' + mode + '_dataset_keystest', 'wb') as fp:
    pickle.dump(keystest, fp)
with open('data/' + mode + '_dataset_keysfinaltest', 'wb') as fp:
    pickle.dump(keysfinaltest, fp)

with open('data/' + mode + 'allblocks', 'wb') as fp:
    pickle.dump(allblocks, fp)