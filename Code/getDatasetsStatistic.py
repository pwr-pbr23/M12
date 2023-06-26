import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import ujson as json
from gensim.models import KeyedVectors
from imblearn.metrics import geometric_mean_score
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import GRU, Dense
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, \
    balanced_accuracy_score
from sklearn.utils import class_weight
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm import tqdm

import myutils

# default mode / type of vulnerability


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
    # "function_injection",
    "remote_code_execution",
    # "cross_frame_scripting",
    # "csv_injection",
    "redirect",
    # "hijack",
    "command_injection",
    "sql",
    "xsrf",
    "xss",
    # "replay_attack",
    # "unauthorized",
    # "brute_force",
    # "flooding",
    # "remote_code_execution",
    # "formatstring",
    # "session_fixation",
    # "cross_origin",
    # "buffer overflow",
    # "cache",
    # "eval",
    # "csv",
    "path_disclosure",
    # "man-in-the-middle",
    # "smurf",
    # "tampering",
    # "sanitize",
    # "denial",
    # "directory_traversal",
    # "clickjack",
    # "spoof",
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


    def proces_data_for_dataset_statistic(keys, blocks):
        X, y = [], []
        for k in tqdm(keys):
            block = blocks[k]
            code = block[0]
            token = myutils.getTokens(code)
            vectorlist = [word_vectors[t].tolist() for t in token if t in word_vectors.key_to_index and t != " "]
            X.append(vectorlist)
            y.append(block[1])


        y = np.where(np.array(y) == 0, 1, 0)
        return y




    # Process data
    allblocks, count = process_data(data, step, fulllength)

    # Create sequences
    keys = np.arange(len(allblocks))
    np.random.shuffle(keys)
    # X, y = create_sequences(keys, allblocks)
    # X = X.reshape(X.shape[0], -1)  # flatten the last two dimensions of X
    y = proces_data_for_dataset_statistic(keys, allblocks)

    # Create dataset statistic and save it to latex table
    print("Dataset statistic:")
    print("Number of samples: ", len(y))
    print("Number of positive samples: ", np.sum(y))
    print("Number of negative samples: ", len(y) - np.sum(y))
    print("Percentage of vulnerable samples: ", np.sum(y) / len(y))

    from tabulate import tabulate


    def print_dataset_statistics_to_latex(y):
        # Calculate statistics
        num_samples = len(y)
        num_positive = np.sum(y)
        num_negative = num_samples - num_positive
        prop_vulnerable = num_positive / num_samples

        # Prepare data for the table
        data = [("Number of samples", num_samples),
                ("Number of positive samples", num_positive),
                ("Number of negative samples", num_negative),
                ("Percentage of vulnerable samples", f"{prop_vulnerable * 100}%")]

        # Define table headers
        headers = ["Statistic", "Value"]

        # Convert data to a LaTeX table
        table = tabulate(data, headers, tablefmt="latex")

        # Print the table
        print(table)

        with open(f"table/{mode}_dataset_statistics.tex", "w") as f:
            f.write(table)


    # Call the function
    print_dataset_statistics_to_latex(y)



