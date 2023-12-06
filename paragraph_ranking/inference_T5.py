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
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)


import pickle
import modelling_T5
import eval
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 't5-large'
save_path = './saved_models/t5-large-finetuned'
tokenizer = AutoTokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
t5_model.to(device)

data_path = "../data/paragraph_ranking/QuoteR/paragraph_ranking_data/667/"
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

with open(os.path.join(context_path, 'ctxid_to_text.json'), 'r') as content:
    ctxid_to_text = json.load(content)
    
# ## len = 11429
# train_set = dataset[:8000]
# valid_set = dataset[8000:9500]
# test_set = dataset[9500:9510]

# print(dataset)

true_labels = {}
## qid, [positive], [cand]
for items in dataset:
    true_labels[str(items[0])] = items[1]

print(true_labels)
print("Evaluating:\n")
t5_model.eval()

t5_model.load_state_dict(torch.load(save_path))

dataset = dataset[0]
qid, label, cands = dataset
# Map question id to text
q_text = ctxid_to_text[str(qid)]

for docid in cands:
    if str(docid) in docid_to_text.keys():
        # Map the docid to text
        ans_text = docid_to_text[str(docid)]

        context = f"Context: {q_text} Document: {ans_text} Relevant: "
        encoded_seq = tokenizer.encode_plus(context,
                                            padding="max_length",
                                            max_length=512,
                                            truncation=True,
                                            return_tensors="pt")
        # Get parameters
        input_id = encoded_seq['input_ids'].to(device)
        attention_mask = encoded_seq['attention_mask'].to(device)

        # outputs = t5_model(**encoded_seq.to(device))
        outputs = t5_model.generate(input_ids=input_id, attention_mask=attention_mask, max_new_tokens=1,
                                    output_scores=True, return_dict_in_generate=True,)


        # print(outputs)

        transition_scores = t5_model.compute_transition_scores(

            outputs.sequences, outputs.scores, normalize_logits=False

        )
        input_length = 1 if t5_model.config.is_encoder_decoder else input_id.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        for tok, score in zip(generated_tokens[0], transition_scores[0]):

            # | token | token string | logits | probability

            print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.detach().cpu().numpy():.3f} | {np.exp(score.detach().cpu().numpy()):.2%}")
            if tokenizer.decode(tok) == 'true':
                print(docid)

        # break
        # beam_outputs = t5_model.generate(
        #     input_ids=input_id, attention_mask=attention_mask,
        #     max_length=2,
        #     early_stopping=True,
        #     num_beams=4,
        #     num_return_sequences=3,
        #     no_repeat_ngram_size=2,
        #     output_scores=True, 
        #     return_dict_in_generate=True
        # )

        # print(beam_outputs)
        # for beam_output in beam_outputs:
        #     print(beam_output)
            # sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            # print (sent)


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