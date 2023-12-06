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

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import pickle
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
from torch.nn import functional as f
import eval

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

model_name = 't5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)


data_path = "../data/paragraph_ranking/QuoteR/"

# Dictionary mapping docid and qid to raw text
with open(os.path.join(data_path, 'docid_to_text.json'), 'r') as content:
    docid_to_text = json.load(content)
    
# with open(os.path.join(data_path, 'ctxid_to_text.json'), 'r') as content:
#     qid_to_text = json.load(content)

## Ctx from QuoteR
with open(os.path.join(data_path, 'qid_to_text.pkl'), 'rb') as content:
    qid_to_text = pickle.load(content)


def get_input_data(dataset, tokenizer, max_seq_len):
    """Creates input parameters for training and validation.
    -----------------
    Arguements:
        dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
    """
    input_ids = []
    attention_masks = []
    lm_labels = []
    decoder_attention_masks = []

    examples = []

    for i, seq in enumerate(tqdm(dataset)):
        qid, ans_labels, cands = seq[0], seq[1], seq[2]
        # Map question id to text
        q_text = qid_to_text[qid]

        # For each answer in the candidates
        for docid in cands:
              # Map the docid to{ text
              ans_text = docid_to_text[str(docid)]

              # If an answer is in the list of relevant answers assign
              # Encode the sequence using BERT tokenizer
              context = f"Context: {q_text} Document: {ans_text} Relevant: "
              encoded_seq = tokenizer.encode_plus(context,
                                                  padding="max_length",
                                                  max_length=max_seq_len,
                                                  truncation=True,
                                                  return_tensors="pt")
              # Get parameters
              input_id = encoded_seq['input_ids'][0]
              attention_mask = encoded_seq['attention_mask'][0]


              # If an answer is in the list of relevant answers assign
              # positive label
              if docid in ans_labels:
                  label = "true"
              else:
                  label = "false"

              tokenized_output = tokenizer.encode_plus(label, max_length=4, pad_to_max_length=True, return_tensors="pt")

              # Get output parameters
              lm_label= tokenized_output["input_ids"][0]
              decoder_attention_mask=  tokenized_output["attention_mask"][0]

              # Each parameter list has the length of the max_seq_len
              assert len(input_id) == max_seq_len, f"Input id dimension incorrect! {len(input_id)} Expected: {max_seq_len}"
              assert len(attention_mask) == max_seq_len, f"Attention mask dimension incorrect! {len(attention_mask)} Expected: {max_seq_len}"

              input_ids.append(input_id)
              attention_masks.append(attention_mask)
              lm_labels.append(lm_label)
              decoder_attention_masks.append(decoder_attention_mask)




    #           # positive label
    #           if docid in ans_labels:
    #               examples.append((f"Context: {q_text} Document: {ans_text} Relevant: ", "true"))
    #           else:
    #               examples.append((f"Context: {q_text} Document: {ans_text} Relevant: ", "false"))

    # random.shuffle(examples)
    # print([lab for (_, lab) in examples][:10])
    # tokenized_inp = tokenizer.encode([ctx for (ctx, _ ) in examples],  max_length=max_seq_len, pad_to_max_length=True, return_tensors="pt")
    # tokenized_output = tokenizer.encode([lab for (_, lab) in examples], max_length=4, pad_to_max_length=True, return_tensors="pt")

    # input_ids  = tokenized_inp["input_ids"].to(device)
    # attention_mask = tokenized_inp["attention_mask"].to(device)

    # lm_labels= tokenized_output["input_ids"].to(device)
    # decoder_attention_mask=  tokenized_output["attention_mask"].to(device)

    return input_ids, attention_masks, lm_labels, decoder_attention_masks


def get_dataloader(dataset, type, tokenizer, max_seq_len, batch_size):
    """Creates train and validation DataLoaders
    """
    input_ids, attention_masks, \
    lm_labels, decoder_attention_masks = get_input_data(dataset, tokenizer, max_seq_len)

    # Convert all inputs to torch tensors
    input_ids = torch.stack([input_id for input_id in input_ids])
    attention_masks = torch.stack([attention_mask for attention_mask in attention_masks])
    lm_labels = torch.stack([lm_label for lm_label in lm_labels])
    decoder_attention_masks = torch.stack([decoder_attention_mask for decoder_attention_mask in decoder_attention_masks])

    # Create the DataLoader for our training set.
    data = TensorDataset(input_ids, attention_masks, lm_labels, decoder_attention_masks)
    if type == "train":
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return dataloader

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
        b_attention_mask = batch[1].to(device)
        b_lm_labels = batch[2].to(device)
        b_decoder_attention_mask = batch[3].to(device)

        # Zero the gradients
        model.zero_grad()
        # Forward pass: the model will return the loss and the logits
        outputs = model(b_input_ids,
                        attention_mask = b_attention_mask,
                        labels = b_lm_labels,
                        decoder_attention_mask= b_decoder_attention_mask)

        # Get loss and predictions
        loss = outputs[0]
        logits = outputs[1]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_lm_labels.to('cpu').numpy()

        # print(label_ids)
        # print(logits)

        # # Calculate the accuracy for a batch
        # tmp_accuracy = eval.get_accuracy(logits, label_ids)

        # # Accumulate the total accuracy.
        # train_accuracy += tmp_accuracy

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

        optimizer.zero_grad()

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
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_lm_labels = batch[2].to(device)
        b_decoder_attention_mask = batch[3].to(device)
        # Don't to compute or store gradients
        with torch.no_grad():
            outputs = model(b_input_ids,
                            attention_mask = b_attention_mask,
                            labels = b_lm_labels,
                            decoder_attention_mask= b_decoder_attention_mask)
        # Get loss and logits
        loss = outputs[0]
        # loss = loss_fn(outputs[1], outputs[1], b_labels)
        logits = outputs[1]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_lm_labels.to('cpu').numpy()

        # # Calculate the accuracy for this batch of test sentences.
        # tmp_eval_accuracy = eval.get_accuracy(logits, label_ids)

        # # Accumulate the total accuracy.
        # eval_accuracy += tmp_eval_accuracy

        # Track the number of steps
        num_steps += 1

        total_loss += loss.item()

    # Calculate loss and accuracy
    avg_loss = total_loss / len(validation_dataloader)
    avg_acc = eval_accuracy/num_steps

    return avg_loss, avg_acc