import os
from datetime import datetime

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import ujson as json
from gensim.models import KeyedVectors
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, \
    balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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
    X, y = create_sequences_bow(keys, allblocks)
    print(f"Rozmiar X po paddingu: {X.shape}")
    print("Mean:", np.mean(X))
    print("Standard deviation:", np.std(X))
    print("Dataset size:", len(y))
    print("Minority class:", np.unique(y, return_counts=True)[1][1])
    print("Majority class:", np.unique(y, return_counts=True)[1][0])
    print("Percentage of minority class: ", np.unique(y, return_counts=True)[1][1] / len(y) * 100)

    # Utwórz model
    models = {
        # 'GNB': GaussianNB(),
        'KNN': KNeighborsClassifier(metric='manhattan'),
        # 'GradientBoost': GradientBoostingClassifier(),
        # 'AdaBoost': AdaBoostClassifier(),
        'BalancedBagging': BalancedBaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=100),
        # 'CART': DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2),


    }
    metrics = {
        'MCC': matthews_corrcoef,
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1': f1_score,
        'Balanced Accuracy': balanced_accuracy_score,
        'G-mean': geometric_mean_score
    }

    preprocessing = {
        'StandardScaler': StandardScaler(),
        'PCA': PCA(n_components=100),
        'RandomUnderSampler': RandomUnderSampler(random_state=0),
        'SMOTE': SMOTE(random_state=0),
    }

    # Log rozpoczęcia uczenia modelu
    print("Rozpoczęcie uczenia modelu...")

    # Ustal liczbę podziałów
    n_splits = 5

    # Utwórz obiekt StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Create the StandardScaler and PCA objects
    scaler = StandardScaler()
    pca = PCA(n_components=100)
    scores = np.zeros((len(models), len(metrics), n_splits))

    # Przeprowadź walidację krzyżową
    for fold_idx, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), total=n_splits):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        # X_train, y_train = rus.fit_resample(X_train, y_train)

        for model_idx, model_name in enumerate(models):
            clf = models[model_name]
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            # Create the confusion matrix
            conf_mat = confusion_matrix(y_test, predictions)

            # Create a heatmap from the confusion matrix
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_mat, annot=True, fmt='d')

            # Set the title, x-label, and y-label
            plt.title(f'Confusion matrix for {model_name} on {mode} data')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Save the confusion matrix plot to a file
            plt.savefig(f'fig/simple_model_{model_name}_{mode}_confusion_matrix.png')

            # Clear the current figure for the next loop
            plt.clf()

            # Zapisz model do pliku
            dump(clf, f'new_models/{model_name}_{mode}.joblib')

            for metric_idx, metric_name in enumerate(metrics):
                metric = metrics[metric_name]
                scores[model_idx, metric_idx, fold_idx] = metric(y_test, predictions)

    average_scores = np.mean(scores, axis=2)


    # Wyświetl średnie wyniki
    scores_df = pd.DataFrame(average_scores, columns=list(metrics.keys()), index=list(models.keys()))
    scores_df = scores_df.round(2)
    print(scores_df)

    # Save to LaTeX
    latex_code = scores_df.style.to_latex(caption=f"Results for {mode} dataset")

    # Write the LaTeX code to a .tex file in the "table" directory
    with open('table/simple_clf_evaluate' + mode + '.tex', 'w') as f:
        f.write(latex_code)

    # Log zakończenia uczenia modelu
    print("Zakończono uczenie modelu.")


