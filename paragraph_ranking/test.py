from curses import OK
from genericpath import exists
import pickle
import os
import json
from random import shuffle

data_path = "../data/paragraph_ranking/QuoteR/paragraph_ranking_data/book_id_12/"
context_path = "../data/paragraph_ranking/QuoteR/"

# Dictionary mapping docid and qid to raw text
with open(os.path.join(data_path, 'docid_to_text_test.json'), 'r') as content:
    docid_to_text = json.load(content)
    
with open(os.path.join(context_path, 'qid_to_text.pkl'), 'rb') as content:
    qid_to_text = pickle.load(content)

with open(os.path.join(data_path, 'dataset_test.pkl'), 'rb') as content:
    dataset = pickle.load(content)


# # printing first 10 ctx in test set
# sample_dataset = [items[0] for items in dataset]
# for idx, sample in enumerate(sample_dataset):
#     print(f"Quote_id: {sample}")
#     print(f"{idx+1}: {qid_to_text[sample]}")

data_path = "../data/paragraph_ranking/Gandhi/full_system_evaluation_data/volume23_book_13.txt/"
context_path = "../data/paragraph_ranking/Gandhi/"


with open(os.path.join(data_path, 'full_system_data_volume23_book_13.txt.json'), 'r') as content:
    full_system_data = json.load(content)

# Dictionary mapping docid and qid to raw text
with open(os.path.join(data_path, 'docid_to_text_volume23_book_13.txt.json'), 'r') as content:
    docid_to_text = json.load(content)

with open(os.path.join(context_path, 'ctxid_to_text.json'), 'r') as content:
    ctxid_to_text = json.load(content)

candidate_ids = [int(doc_id) for doc_id in docid_to_text.keys()]


dataset = []
for ctx_id, items in full_system_data.items():
    shuffle(candidate_ids)
    dataset.append([int(ctx_id), [items[1]], candidate_ids])

data_path = "../data/paragraph_ranking/Gandhi/paragraph_ranking_data/volume23_book_13/"
context_path = "../data/paragraph_ranking/Gandhi/"

os.makedirs(data_path, exist_ok=True)

with open(os.path.join(data_path, 'dataset_test.pkl'), 'wb') as content:
    pickle.dump(dataset, content)

with open(os.path.join(data_path, 'docid_to_text_test.json'), 'w') as content:
    json.dump(docid_to_text, content)

print(dataset)