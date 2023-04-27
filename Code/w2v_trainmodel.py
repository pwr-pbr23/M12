import nltk
from gensim.models import Word2Vec
import os.path
import pickle
import sys

mode = "withString"  # default
if len(sys.argv) > 1:
    mode = sys.argv[1]

# Loading the training corpus`


print(f"Loading {mode}")
with open(f"w2v/pythontraining_{mode}_X", "r") as file:
    pythondata = file.read().lower().replace("\n", " ")

print(f"Length of the training file: {len(pythondata)}.")
print(f"It contains {pythondata.count(' ')} individual code tokens.")

# Preparing the dataset (or loading already processed dataset to not do everything again)
processed_fname = f"data/pythontraining_processed_{mode}"
if os.path.isfile(processed_fname):
    with open(processed_fname, "rb") as fp:
        all_words = pickle.load(fp)
    print("loaded processed model.")
else:
    print("now processing...")
    all_sentences = nltk.sent_tokenize(pythondata)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    print("saving")
    with open(processed_fname, "wb") as fp:
        pickle.dump(all_words, fp)

print("processed.\n")

# trying out different parameters
min_counts = [10, 30, 50, 100, 300, 500, 5000]
iterations = [1, 5, 10, 30, 50, 100]
sizes = [5, 10, 15, 30, 50, 75, 100, 200, 300]

for min_count in min_counts:
    for iteration in iterations:
        for size in sizes:

            fname = f"w2v/word2vec_{mode}{min_count}-{iteration}-{size}.model"

            if os.path.isfile(fname):
                print("model already exists.")
                continue

            print(f"\n{mode} W2V model with min count {min_count} and {iteration} iterations and size {size}")
            print("calculating model...")
            # training the model
            model = Word2Vec(all_words, vector_size=size, min_count=min_count, epochs=iteration, workers=4)
            vocabulary = model.wv.key_to_index

            # saving the model
            model.save(fname)