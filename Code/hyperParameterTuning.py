import os
import sys
from datetime import datetime

import numpy as np
import ujson as json
from gensim.models import KeyedVectors
from keras.utils import pad_sequences
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from tqdm import tqdm

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


# Process data
allblocks, count = process_data(data, step, fulllength)

# Create sequences
keys = np.arange(len(allblocks))
np.random.shuffle(keys)
X, y = create_sequences(keys, allblocks)
X = X.reshape(X.shape[0], -1)  # flatten the last two dimensions of X
print(f"Rozmiar X po paddingu: {X.shape}")
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
print(type(class_weights))
class_weights = dict(enumerate(class_weights))

# # Parametry do tuningu
# parameters = {
#     'max_depth': [None, 5, 10, 15, 20],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 2, 5, 10],
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'class_weight': [None, class_weights]
# }
# Definiowanie pipeline'u
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Dodajemy StandardScaler jako pierwszy krok w pipeline'ie
    ('pca', PCA()),  # Dodajemy PCA jako drugi krok
    ('clf', KNeighborsClassifier())  # Na końcu dodajemy klasyfikator
])

# Definiowanie siatki parametrów do strojenia
parameters = {
    'pca__n_components': [None, 20, 50, 100],  # Liczba głównych komponentów do zachowania przez PCA
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5],
    'clf__criterion': ['gini', 'entropy'],
    'clf__class_weight': [None, class_weights],
    'clf__random_state': [42],
    'clf__ccp_alpha': [0.0, 0.2],

}

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Strojenie hiperparametrów
clf = GridSearchCV(pipe, parameters, cv=5, verbose=2)
clf.fit(X_train, y_train)

# Wyświetlenie najlepszych parametrów
print(f"Najlepsze parametry: {clf.best_params_}")

# Wyświetlenie najlepszego wyniku
print(f"Najlepszy wynik: {clf.best_score_}")