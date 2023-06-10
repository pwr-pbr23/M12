from concurrent.futures import ThreadPoolExecutor
import myutils
import sys
import os.path
import ujson as json
from datetime import datetime
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
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

def split_dict_equally(input_dict, chunks):
    "Splits dict by keys. Returns a list of dictionaries."
    # prep with empty dicts
    return_list = [dict() for idx in range(chunks)]
    idx = 0
    for k,v in input_dict.items():
        return_list[idx][k] = v
        if idx < chunks-1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list


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

    y = np.where(np.array(y) == 0, 1, 0)
    return X, y


# Split data into chunks for parallel processing
chunks = split_dict_equally(data, 32)

# First apply ThreadPoolExecutor to process_data
with ThreadPoolExecutor(max_workers=32) as executor:
    futures_process_data = [executor.submit(process_data, chunk, step, fulllength) for chunk in chunks]

# Wait for all futures to finish and collect their results
results_process_data = [future.result() for future in futures_process_data]

# Concatenate all results together
allblocks = [block for result in results_process_data for block in result[0]]
count = sum(result[1] for result in results_process_data)

keys = np.arange(len(allblocks))
np.random.shuffle(keys)

cutoff = round(0.7 * len(keys))
cutoff2 = round(0.85 * len(keys))
keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]

# Split keys into chunks for parallel processing
key_chunks = np.array_split(keys, 32)

# Now apply ThreadPoolExecutor to create_sequences
with ThreadPoolExecutor(max_workers=32) as executor:
    futures_create_sequences = [executor.submit(create_sequences, key_chunk, allblocks) for key_chunk in key_chunks]

# Wait for all futures to finish and collect their results
results_create_sequences = [future.result() for future in futures_create_sequences]

# Concatenate all results together
X = [x for result in results_create_sequences for x in result[0]]
y = [y for result in results_create_sequences for y in result[1]]

# Convert data to pandas DataFrame
df = pd.DataFrame({'X': X, 'y': y})

# Define the path to the CSV file
csv_path = f'csv/{mode}+dataset.csv'

# Save DataFrame to CSV
df.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")
