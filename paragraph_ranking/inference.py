from tqdm import tqdm
import numpy as np
import random
import torch
import json
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from rank_bm25 import BM25Okapi
import math

import pickle
import modelling
import eval
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = "../data/paragraph_ranking/QuoteR/paragraph_ranking_data/27889/"
context_path = "../data/paragraph_ranking/QuoteR/paragraph_ranking_data"

# data_path = "../data/paragraph_ranking/Gandhi/paragraph_ranking_data/volume23_book_13/"
# context_path = "../data/paragraph_ranking/Gandhi/"

with open(os.path.join(data_path, 'dataset.pkl'), 'rb') as content:
    dataset = pickle.load(content)

# Dictionary mapping docid and qid to raw text (Should be test set)
with open(os.path.join(data_path, 'docid_to_text.json'), 'r') as content:
    docid_to_text = json.load(content)

# Ctx from QuoteR
# with open(os.path.join(context_path, 'qid_to_text.pkl'), 'rb') as content:
#     ctxid_to_text = pickle.load(content)

with open(os.path.join(context_path, 'ctxid_to_text_paraphrased.json'), 'r') as content:
    ctxid_to_text = json.load(content)
    
# ## len = 11429
# train_set = dataset[:8000]
# valid_set = dataset[8000:9500]
# test_set = dataset[9500:9510]

print(dataset)

true_labels = {}
## qid, [positive], [cand]
for items in dataset:
    true_labels[str(items[0])] = items[1]

print("Evaluating:\n")
# Load model
model_path = 'bert-base-uncased'

# Load BertForSequenceClassification - pretrained BERT model 
# with a single linear classification layer on top
model = BertForSequenceClassification.from_pretrained(model_path, cache_dir=None, num_labels=2)
print('\nLoading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

model.to(device)
trained_model_path = 'saved_models/paragraph_ranking_9_neg.pt'
model.load_state_dict(torch.load(trained_model_path))


def rank_paragraph_bm25(test_set, docid_to_text):

    text_to_docid = {contents: docid for docid, contents in  docid_to_text.items()}
        
    corpus = list(text_to_docid.keys())
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_ranked_list = {}
    for i in range(len(test_set)):
        seq = test_set[i]
        qid, label, cands = seq[0], seq[1], seq[2]
        q_text = ctxid_to_text[qid]
        tokenized_query = q_text.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)

        x = 0
        docid_to_scores = {}
        for docid, text in docid_to_text.items():
            docid_to_scores[docid] = doc_scores[x]
            x+=1
        docid_to_scores = {id: score for id, score in enumerate(doc_scores)}

        sorted_docid_to_scores = dict(sorted(docid_to_scores.items(), key=lambda item: item[1], reverse=True))

        bm25_ranked_list[qid] = list(map(int, sorted_docid_to_scores.keys()))

    return bm25_ranked_list

def predict(model, q_text, cands, max_seq_len):
    """Re-ranks the candidates answers for each question.

    Returns:
        ranked_ans: list of re-ranked candidate docids
        sorted_scores: list of relevancy scores of the answers
    -------------------
    Arguments:
        model - PyTorch model
        q_text - str - query
        cands -List of retrieved candidate docids
        max_seq_len - int
    """
    # Convert list to numpy array
    cands_id = np.array(cands)
    # Empty list for the probability scores of relevancy
    scores = []
    # For each answer in the candidates
    for docid in cands:
        if str(docid) in docid_to_text.keys():
          # Map the docid to text
          ans_text = docid_to_text[str(docid)]
          # Create inputs for the model
          encoded_seq = tokenizer.encode_plus(q_text, ans_text,
                                              max_length=max_seq_len,
                                              pad_to_max_length=True,
                                              return_token_type_ids=True,
                                              return_attention_mask = True)

          # Numericalized, padded, clipped seq with special tokens
          input_ids = torch.tensor([encoded_seq['input_ids']]).to(device)
          # Specify question seq and answer seq
          token_type_ids = torch.tensor([encoded_seq['token_type_ids']]).to(device)
          # Sepecify which position is part of the seq which is padded
          att_mask = torch.tensor([encoded_seq['attention_mask']]).to(device)
          # Don't calculate gradients
          with torch.no_grad():
              # Forward pass, calculate logit predictions for each QA pair
              outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=att_mask)
          # Get the predictions
          logits = outputs[0]
          # Apply activation function
          pred = softmax(logits, dim=1)
          # Move logits and labels to CPU
          pred = pred.detach().cpu().numpy()
          # Append relevant scores to list (where label = 1)
          scores.append(pred[:,1][0])
          # Get the indices of the sorted similarity scores
          sorted_index = np.argsort(scores)[::-1]
          # Get the list of docid from the sorted indices
          ranked_ans = list(cands_id[sorted_index])
          sorted_scores = list(np.around(sorted(scores, reverse=True),decimals=3))
        

    return ranked_ans, sorted_scores


def get_rank(model, test_set, max_seq_len):
    """Re-ranks the candidates answers for each question.

    Returns:
        qid_pred_rank: Dictionary
            key - qid
            value - list of re-ranked candidates
    -------------------
    Arguments:
        model - PyTorch model
        test_set  List of lists
        max_seq_len - int
    """
    # Initiate empty dictionary
    qid_pred_rank = {}
    # Set model to evaluation mode
    model.eval()
    # For each element in the test set
    for i, seq in enumerate(tqdm(test_set)):
        # question id, list of rel answers, list of candidates
        qid, label, cands = seq[0], seq[1], seq[2]
        # Map question id to text
        q_text = ctxid_to_text[str(qid)]

        # List of re-ranked docids and the corresponding probabilities
        ranked_ans, sorted_scores = predict(model, q_text, cands, max_seq_len)

        # Dict - key: qid, value: ranked list of docids
        qid_pred_rank[qid] = ranked_ans

    return qid_pred_rank

# Get Rank using bm25
# qid_pred_rank = utils.rank_paragraph_bm25(dataset, docid_to_text, ctxid_to_text)
# print(qid_pred_rank)

# # Get Rank using sentence-bert
qid_pred_rank = utils.rank_paragraph_sbert(dataset, docid_to_text, ctxid_to_text, device)
# print(qid_pred_rank)

# # # Get rank using BERT reranking
# qid_pred_rank = get_rank(model, dataset, 512)



k = 3
num_q = len(dataset)


# print(qid_pred_rank)
filtered_labels = { int(your_key): true_labels[str(your_key)] for your_key in true_labels.keys() }


k = 9
num_q = len(qid_pred_rank)
# Evaluate
# MRR, average_ndcg, precision, rank_pos = evaluate(qid_pred_rank, filtered_labels, k)



mAP = eval.mAP_k(filtered_labels, qid_pred_rank, k=500)

# print("\n\nAverage nDCG@{0} for {1} queries: {2:.3f}".format(k, num_q, average_ndcg))
# print("MRR@{0} for {1} queries: {2:.3f}".format(k, num_q, MRR))
# print("Average Precision@1 for {0} queries: {1:.3f}".format(num_q, precision))
print("MAP for {0} queries: {1:.3f}".format(num_q, mAP))
print("Acc@1 : {0:.3f}".format(eval.top_k_acc(filtered_labels, qid_pred_rank, k=1)))
print("Acc@3 : {0:.3f}".format(eval.top_k_acc(filtered_labels, qid_pred_rank, k=3)))
print("Acc@10 : {0:.3f}".format(eval.top_k_acc(filtered_labels, qid_pred_rank, k=10)))
print("Acc@100 : {0:.3f}".format(eval.top_k_acc(filtered_labels, qid_pred_rank, k=100)))
print("Acc@200 : {0:.3f}".format(eval.top_k_acc(filtered_labels, qid_pred_rank, k=200)))