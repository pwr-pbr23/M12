import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
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


    # Określamy rozmiary wejść
    input_shape = X_train.shape[1:]
    print(input_shape)

    # Tworzymy model
    model = Sequential()
    model.add(GRU(50, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))

    # Kompilujemy model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[myutils.mcc])

    # account with class_weights for the class-imbalanced nature of the underlying data
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print(type(class_weights))
    class_weights = dict(enumerate(class_weights))
    print(type(class_weights))
    print(class_weights)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Define checkpoint to save best model
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Trenujemy model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                        workers=8, use_multiprocessing=True, callbacks=[early_stopping, model_checkpoint])

    # Plotting MCC metric
    plt.plot(history.history['mcc'], marker='o')
    plt.plot(history.history['val_mcc'], marker='o')
    plt.title('Model MCC')
    plt.ylabel('MCC')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('mcc_plot.png')  # saves the plot to a file
    plt.savefig(f'fig/GRU_{mode}_mcc_plot.png')  # saves the plot to a file
    plt.clf()

    # Plotting loss
    plt.plot(history.history['loss'], marker='o')
    plt.plot(history.history['val_loss'], marker='o')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss_plot.png')  # saves the plot to a file
    plt.savefig(f'fig/GRU_{mode}_loss_plot.png')  # saves the plot to a file
    plt.clf()


    metrics = {
        'MCC': matthews_corrcoef,
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1': f1_score,
        'Balanced Accuracy': balanced_accuracy_score,
        'G-mean': geometric_mean_score
    }

    # Evaluate the model

    # Przewidujemy klasy na zbiorze testowym
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).flatten()  # Zaokrąglenie przewidywań i spłaszczenie do 1D

    # Inicjalizacja słownika na wyniki
    results = {}

    # Obliczanie i drukowanie metryk
    for metric_name, metric_func in metrics.items():
        result = metric_func(y_test, y_pred)

        results[metric_name] = round(result, 2)
        print(f"{metric_name}: {result}")

    # Przykładowo, wyniki można zapisać jako dataframe pandas
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])

    print(results_df)

    latex_code = results_df.style.to_latex(caption=f"Results for {mode} dataset")

    # Write the LaTeX code to a .tex file in the "table" directory
    with open('table/GRU_' + mode + '.tex', 'w') as f:
        f.write(latex_code)

    now = datetime.now()  # current date and time
    nowformat = now.strftime("%H:%M")
    print("saving GRU model " + mode + ". ", nowformat)
    model.save('GRUModel/GRU_model_' + mode + '.h5')  # creates a HDF5 file 'my_model.h5'
    print(" \n\n")

