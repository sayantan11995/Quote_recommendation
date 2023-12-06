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
import modelling

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# from pyserini.search import pysearch

# Setting device on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

config = {'bert_model_name': 'bert-qa',
          'max_seq_len': 512,
          'batch_size': 4,
          'learning_rate': 2e-5,
          'weight_decay': 0.01,
          'n_epochs': 10,
          'num_warmup_steps': 1000}

data_path = "../data/paragraph_ranking/QuoteR/"

with open(os.path.join(data_path, 'dataset_9_neg.pkl'), 'rb') as content:
    dataset = pickle.load(content)
    
## len = 11429
train_set = dataset[:8000]
valid_set = dataset[8000:9500]
test_set = dataset[9500:]

# Labels
with open(os.path.join(data_path, 'labels_9_neg.pkl'), 'rb') as content:
    labels = pickle.load(content)

model_path = 'bert-base-uncased'
# Load the BERT tokenizer.
print('\nLoading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)


# Get dataloaders
train_dataloader = modelling.get_dataloader(train_set, 'train', 
                                  config['max_seq_len'], 
                                  config['batch_size'])
validation_dataloader = modelling.get_dataloader(valid_set, 'validation', 
                                       config['max_seq_len'], 
                                       config['batch_size'])

print("\n\nSize of the training DataLoader: {}".format(len(train_dataloader)))
print("Size of the validation DataLoader: {}".format(len(validation_dataloader)))


model_path = 'bert-base-uncased'

# Load BertForSequenceClassification - pretrained BERT model 
# with a single linear classification layer on top
model = BertForSequenceClassification.from_pretrained(model_path, cache_dir=None, num_labels=2)

model.to(device)

optimizer = AdamW(model.parameters(), 
                  lr = config['learning_rate'], 
                  weight_decay = config['weight_decay'])

n_epochs = config['n_epochs']


# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * n_epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = config['num_warmup_steps'],
                                            num_training_steps = total_steps)

# Lowest validation lost
best_valid_loss = float('inf')

for epoch in range(n_epochs):
    # Evaluate training loss
    train_loss, train_acc = modelling.train(model, train_dataloader, optimizer, scheduler)
    # Evaluate validation loss
    valid_loss, valid_acc = modelling.validate(model, validation_dataloader)
    # At each epoch, if the validation loss is the best
    if valid_loss < best_valid_loss:
        
        best_valid_loss =  valid_loss
    print("\n\n Epoch {}:".format(epoch+1))
    print("\t Train Loss: {} | Train Accuracy: {}%".format(round(train_loss, 3), round(train_acc*100, 2)))
    print("\t Validation Loss: {} | Validation Accuracy: {}%\n".format(round(valid_loss, 3), round(valid_acc*100, 2)))


torch.save(model.state_dict(), 'saved_models/paragraph_ranking_9_neg.pt')
