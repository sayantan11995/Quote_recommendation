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

import pickle
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
from torch.nn import functional as f

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# from pyserini.search import pysearch

# Setting device on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


data_path = "../data/paragraph_ranking/QuoteR/"

# Dictionary mapping docid and qid to raw text
with open(os.path.join(data_path, 'docid_to_text.json'), 'r') as content:
    docid_to_text = json.load(content)
    
# with open(os.path.join(data_path, 'ctxid_to_text.json'), 'r') as content:
#     qid_to_text = json.load(content)

## Ctx from QuoteR
with open(os.path.join(data_path, 'qid_to_text.pkl'), 'rb') as content:
    qid_to_text = pickle.load(content)

model_path = 'bert-base-uncased'
# Load the BERT tokenizer.
print('\nLoading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

def get_input_data(dataset, max_seq_len):
    """Creates input parameters for training and validation.

    Returns:
        input_ids: List of lists
                Each element contains a list of padded/truncated numericalized
                tokens of the sequences including [CLS] and [SEP] tokens
                e.g. [[101, 2054, 2003, 102, 2449, 1029, 102], ...]
        token_type_ids: List of lists
                Each element contains a list of segment token indices to
                indicate first (question) and second (answer) parts of the inputs.
                0 corresponds to a question token, 1 corresponds an answer token
                e.g. [[0, 0, 0, 0, 1, 1, 1], ...]
        att_masks: List of lists
                Each element contains a list of mask values to avoid
                performing attention on padding token indices.
                1 for tokens that are NOT MASKED, 0 for MASKED tokens.
                e.g. [[1, 1, 1, 1, 1, 1, 1], ...]
        labels: List of 1's and 0's incidating relevacy of answer
    -----------------
    Arguements:
        dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
    """
    input_ids = []
    token_type_ids = []
    att_masks = []
    labels = []

    for i, seq in enumerate(tqdm(dataset)):
        qid, ans_labels, cands = seq[0], seq[1], seq[2]
        # Map question id to text
        q_text = qid_to_text[qid]
        # For each answer in the candidates
        for docid in cands:
              # Map the docid to{ text
              ans_text = docid_to_text[str(docid)]
              # Encode the sequence using BERT tokenizer
              encoded_seq = tokenizer.encode_plus(q_text, ans_text,
                                                  padding="max_length",
                                                  max_length=max_seq_len,
                                                  truncation=True,
                                                  return_token_type_ids=True,
                                                  return_attention_mask = True)
              # Get parameters
              input_id = encoded_seq['input_ids']
              token_type_id = encoded_seq['token_type_ids']
              att_mask = encoded_seq['attention_mask']

              print(encoded_seq)

              # If an answer is in the list of relevant answers assign
              # positive label
              if docid in ans_labels:
                  label = 1
              else:
                  label = 0

              # Each parameter list has the length of the max_seq_len
              assert len(input_id) == max_seq_len, "Input id dimension incorrect!"
              assert len(token_type_id) == max_seq_len, "Token type id dimension incorrect!"
              assert len(att_mask) == max_seq_len, "Attention mask dimension incorrect!"

              input_ids.append(input_id)
              token_type_ids.append(token_type_id)
              att_masks.append(att_mask)
              labels.append(label)
    return input_ids, token_type_ids, att_masks, labels


def get_dataloader(dataset, type, max_seq_len, batch_size):
    """Creates train and validation DataLoaders with input_ids,
    token_type_ids, att_masks, and labels

    Returns:
        train_dataloader: DataLoader object
        validation_dataloader: DataLoader object

    -----------------
    Arguements:
        dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
        type: str - 'train' or 'validation'
        max_seq_len: int
        batch_size: int
    """
    input_id, token_type_id, \
    att_mask, label = get_input_data(dataset, max_seq_len)

    # Convert all inputs to torch tensors
    input_ids = torch.tensor(input_id)
    token_type_ids = torch.tensor(token_type_id)
    att_masks = torch.tensor(att_mask)
    labels = torch.tensor(label)

    # Create the DataLoader for our training set.
    data = TensorDataset(input_ids, token_type_ids, att_masks, labels)
    if type == "train":
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return dataloader

def get_accuracy(preds, labels):
    """Compute the accuracy of binary predictions.

    Returns:
        accuracy: float
    -----------------
    Arguments:
        preds: Numpy list with two columns of probabilities for each label
        labels: List of labels
    """
    # Get the label (column) with the higher probability
    predictions = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    # Compute accuracy
    accuracy = np.sum(predictions == labels) / len(labels)

    return accuracy


# loss_fn = CosineEmbeddingLoss()

def train(model, train_dataloader, optimizer, scheduler):
    """Trains the model and returns the average loss and accuracy.

    Returns:
        avg_loss: Float
        avg_acc: Float
    ----------
    Arguements:
        model: Torch model
        train_dataloader: DataLoader object
        optimizer: Optimizer object
        scheduler: Scheduler object
    """
    # Cumulated Training loss and accuracy
    total_loss = 0
    train_accuracy = 0
    # Track the number of steps
    num_steps = 0
    # Set model in train mode
    model.train()
    # For each batch of training data
    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get tensors and move to gpu
        # batch contains four PyTorch tensors:
        #   [0]: input ids
        #   [1]: token_type_ids
        #   [2]: attention masks
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # Zero the gradients
        model.zero_grad()
        # Forward pass: the model will return the loss and the logits
        outputs = model(b_input_ids,
                        token_type_ids = b_token_type_ids,
                        attention_mask = b_input_mask,
                        labels = b_labels)

        # Get loss and predictions
        loss = outputs[0]
        logits = outputs[1]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for a batch
        tmp_accuracy = get_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        train_accuracy += tmp_accuracy

        # Track the number of batches
        num_steps += 1

        # Accumulate the training loss over all of the batches
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update scheduler
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = train_accuracy/num_steps

    return avg_loss, avg_acc


def validate(model, validation_dataloader):
    """Validates the model and returns the average loss and accuracy.

    Returns:
        avg_loss: Float
        avg_acc: Float
    ----------
    Arguements:
        model: Torch model
        validation_dataloader: DataLoader object
    """
    # Set model to evaluation mode
    model.eval()
    # Cumulated Training loss and accuracy
    total_loss = 0
    eval_accuracy = 0
    # Track the number of steps
    num_steps = 0

    # For each batch of the validation data
    for batch in tqdm(validation_dataloader):
        # Move tensors from batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from the dataloader
        b_input_ids, b_token_type_ids, b_input_masks, b_labels = batch
        # Don't to compute or store gradients
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids = b_token_type_ids,
                            attention_mask = b_input_masks,
                            labels= b_labels)
        # Get loss and logits
        loss = outputs[0]
        # loss = loss_fn(outputs[1], outputs[1], b_labels)
        logits = outputs[1]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = get_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of steps
        num_steps += 1

        total_loss += loss.item()

    # Calculate loss and accuracy
    avg_loss = total_loss / len(validation_dataloader)
    avg_acc = eval_accuracy/num_steps

    return avg_loss, avg_acc


