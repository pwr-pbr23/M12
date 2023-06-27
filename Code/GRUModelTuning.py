from keras import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters



import argparse
import os
from datetime import datetime

import pandas as pd
from gensim.models import KeyedVectors
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import pad_sequences
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, \
    balanced_accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from tqdm import tqdm
import sys
import numpy as np
import ujson as json
from joblib import dump
import myutils

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
modes = [
    "function_injection",
    "remote_code_execution",
    # "cross_frame_scripting",
    "csv_injection",
    "redirect",
    "hijack",
    "command_injection",
    "sql",
    "xsrf",
    "xss",
    "replay_attack",
    "unauthorized",
    "brute_force",
    "flooding",
    "remote_code_execution",
    "formatstring",
    "session_fixation",
    "cross_origin",
    # "buffer overflow",
    # "cache",
    "eval",
    "csv",
    # "path_disclosure",
    # "man-in-the-middle",
    "smurf",
    "tampering",
    "sanitize",
    "denial",
    "directory_traversal",
    "clickjack",
    "spoof",
]

for mode in modes:
print(f"starting {mode}")
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
        X.append(vectorlist)
        y.append(block[1])

        # Pad sequences
    X = pad_sequences(X, maxlen=fulllength, padding='pre', truncating='pre', dtype='float32')

    y = np.where(np.array(y) == 0, 1, 0)
    return X, y


def process_X_y(keys, blocks):
    X, y = [], []
    for k in tqdm(keys):
        block = blocks[k]
        code = block[0]
        tokens = myutils.getTokens(code)
        vectorlist = [word_vectors[t] for t in tokens if t in word_vectors.key_to_index and t != " "]
        # Compute the mean of the vectors
        mean_vector = np.mean(vectorlist, axis=0) if vectorlist else np.zeros(word_vectors.vector_size)
        X.append(mean_vector.tolist())
        y.append(block[1])

    X = np.array(X)
    y = np.where(np.array(y) == 0, 1, 0)
    return X, y

def create_sequences_bow(keys, blocks):
    X, y = [], []
    vectorizer = CountVectorizer(tokenizer=myutils.getTokens)

    for k in tqdm(keys):
        block = blocks[k]
        code = block[0]
        X.append(code)
        y.append(block[1])

    X = vectorizer.fit_transform(X)
    y = np.where(np.array(y) == 0, 1, 0)

    return X.toarray(), y


# Process data
allblocks, count = process_data(data, step, fulllength)

# Create sequences
keys = np.arange(len(allblocks))
np.random.shuffle(keys)
# X, y = create_sequences(keys, allblocks)
# X = X.reshape(X.shape[0], -1)  # flatten the last two dimensions of X
X, y = create_sequences(keys, allblocks)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]
def build_model(hp: HyperParameters):
    model = Sequential()
    model.add(GRU(hp.Int('units', min_value=32, max_value=512, step=32),
                  input_shape=input_shape,
                  dropout=hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=[myutils.mcc])

    return model


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # number of different hyperparameter combinations to try
    executions_per_trial=3,  # number of times to train each model, to reduce variance
    directory='random_search',
    project_name='code_vulnerability'
)

tuner.search_space_summary()

# This will start the tuning process
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

# You can then get the best model as follows:
best_model = tuner.get_best_models(num_models=1)[0]

# And the hyperparameters of the best model like this:
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print('Best Hyperparameters:', best_hyperparameters.values)