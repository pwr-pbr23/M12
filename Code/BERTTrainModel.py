from transformers import BertTokenizerFast, BertForMaskedLM, TrainingArguments, Trainer
import torch
import nltk
from tqdm import tqdm
import os.path
import pickle
import sys
from torch.utils.data import Dataset, DataLoader

class PythonCodeDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

mode = "withString"  # default
if len(sys.argv) > 1:
    mode = sys.argv[1]

print(f"Loading {mode}")
# with open(f"w2v/pythontraining_{mode}_X", "r", encoding='utf-8') as file:
#     pythondata = file.read().lower().replace("\n", " ")
#
# print(f"Length of the training file: {len(pythondata)}.")
# print(f"It contains {pythondata.count(' ')} individual code tokens.")

processed_fname = f"data/pythontraining_processed_{mode}"
if os.path.isfile(processed_fname):
    with open(processed_fname, "rb") as fp:
        all_words = pickle.load(fp)
    print("loaded processed model.")
else:
    print("now processing...")
    all_sentences = nltk.sent_tokenize(pythondata)
    all_words = [' '.join(nltk.word_tokenize(sent)) for sent in tqdm(all_sentences, desc="Tokenizing sentences")]
    print("saving")
    with open(processed_fname, "wb") as fp:
        pickle.dump(all_words, fp)

print("processed.\n")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# tokenization
print("Tokenizing...\n")
tokenized_datasets = [tokenizer(sent, padding="max_length", truncation=True) for sent in tqdm(all_words)]


train_dataset = PythonCodeDataset(tokenized_datasets)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

trainer.save_model("./BERT")
