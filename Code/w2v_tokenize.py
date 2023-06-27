import sys

import chardet as chardet
import nltk.tokenize
import time

pythondata = ""
mode = "withString"  # default
# mode = "withoutString"

if len(sys.argv) > 1:
    mode = sys.argv[1]


# use the nltk tokenizer to tokenize the words in the corpus
with open("w2v/pythontraining_edit.txt", "r", encoding="utf-8") as file:
    data = file.read()


tokens = nltk.tokenize.word_tokenize(data)

count = 0
totalcount = 0
comment = 0
part = 0

for token in tokens:
    totalcount += 1
    count += 1
    if totalcount % 1000 == 0:
        print(totalcount)

    if '"""' in token:
        comment += 1
        continue

    if mode == "withoutString":
        if "\"" in token:
            token = "\"string\""

    if token == "\n":
        pythondata += "\n"
    elif token == "\t":
        pythondata += "  "
    else:
        pythondata += " " + token.strip()

    # save in parts to reduce computational load
    if count > 1000000:
        print("saving part " + str(part) + " (" + mode + ") " + str(totalcount))
        with open('w2v/pythontraining' + "_" + mode + "_" + str(part), 'w', encoding="utf-8") as outfile:
            outfile.write(pythondata)
        pythondata = ""
        part += 1
        count = 0

with open('w2v/pythontraining' + "_" + mode + "_" + str(part), 'w', encoding="utf-8") as outfile:
    outfile.write(pythondata)