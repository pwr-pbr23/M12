from keras.utils import pad_sequences

import myutils
import sys
import os.path
import ujson as json
from datetime import datetime
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

import pandas as pd

# default mode / type of vulnerability
mode = "sql"

# get the vulnerability from the command line argument
if (len(sys.argv) > 1):
    mode = sys.argv[1]

progress = 0
# count = 0

### paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step length n in the description
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

# allblocks = []

def process_data(data, step, fulllength):
    allblocks = []
    count = 0

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

    return allblocks, count


def create_sequences(keys, blocks):
    X, y = [], []
    for k in tqdm(keys):
        block = blocks[k]
        code = block[0]
        token = myutils.getTokens(code)
        vectorlist = [word_vectors[t].tolist() for t in token if t in word_vectors.key_to_index and t != " "]
        # Spłaszczenie listy list do pojedynczej listy
        vectorlist_flat = [item for sublist in vectorlist for item in sublist]
        # Konwersja na float
        vectorlist_flat = np.array(vectorlist_flat).astype('float32')
        X.append(vectorlist_flat)
        y.append(block[1])

        # Konwersja y do NumPy array
    y = np.where(np.array(y) == 0, 1, 0)

    # Padding sekwencji do określonej maksymalnej długości
    X = pad_sequences(X, maxlen=fulllength, padding='pre', truncating='pre', dtype='float32')

    return X, y


# Process data
allblocks, count = process_data(data, step, fulllength)

# Create sequences
keys = np.arange(len(allblocks))
np.random.shuffle(keys)
X, y = create_sequences(keys, allblocks)


