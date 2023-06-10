import sys
import pickle
import numpy
from keras.utils import pad_sequences
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model
import myutils

def load_dataset(mode):
    with open(f"data/{mode}_dataset_finaltest_X", "rb") as fp:
        finaltest_x = pickle.load(fp)
    with open(f"data/{mode}_dataset_finaltest_Y", "rb") as fp:
        finaltest_y = pickle.load(fp)

    for i in range(len(finaltest_y)):
        finaltest_y[i] = 1 if finaltest_y[i] == 0 else 0

    max_length = 200
    finaltest_x = pad_sequences(finaltest_x, maxlen=max_length)

    return finaltest_x, finaltest_y


def evaluate_model(mode):
    model = load_model(f"model/LSTM_model_{mode}.h5", custom_objects={"f1_loss": myutils.f1_loss, "f1": myutils.f1})
    X_finaltest, y_finaltest = load_dataset(mode)
    yhat_classes = model.predict_classes(X_finaltest, verbose=0)

    print(f"\n{mode}\n{'-'*50}")
    print(classification_report(y_finaltest, yhat_classes))
    print(f"Accuracy: {accuracy_score(y_finaltest, yhat_classes)}\n{'-'*50}\n")


if __name__ == "__main__":
    modes = ["sql", "xss"]
    for mode in modes:
        evaluate_model(mode)