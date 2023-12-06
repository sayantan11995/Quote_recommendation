from tqdm import tqdm
import numpy as np
import random
import torch
import json
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
import pickle
import modelling_T5

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# from pyserini.search import pysearch

# Setting device on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)

save_path = './saved_models/t5-large-finetuned'

model_name = 't5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
t5_model.to(device)

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

config = {'bert_model_name': 'bert-qa',
          'max_seq_len': 512,
          'batch_size': 4,
          'learning_rate': 2e-5,
          'weight_decay': 0.01,
          'n_epochs': 3,
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


# Get dataloaders
train_dataloader = modelling_T5.get_dataloader(train_set, 'train', tokenizer,
                                  config['max_seq_len'], 
                                  config['batch_size'])
validation_dataloader = modelling_T5.get_dataloader(valid_set, 'validation', tokenizer,
                                       config['max_seq_len'], 
                                       config['batch_size'])

print("\n\nSize of the training DataLoader: {}".format(len(train_dataloader)))
print("Size of the validation DataLoader: {}".format(len(validation_dataloader)))


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
    train_loss, train_acc = modelling_T5.train(t5_model, train_dataloader, optimizer, scheduler)
    # Evaluate validation loss
    valid_loss, valid_acc = modelling_T5.validate(t5_model, validation_dataloader)
    # At each epoch, if the validation loss is the best
    if valid_loss < best_valid_loss:
        
        best_valid_loss =  valid_loss
    print("\n\n Epoch {}:".format(epoch+1))
    print("\t Train Loss: {} | Train Accuracy: {}%".format(round(train_loss, 3), round(train_acc*100, 2)))
    print("\t Validation Loss: {} | Validation Accuracy: {}%\n".format(round(valid_loss, 3), round(valid_acc*100, 2)))


torch.save(t5_model.state_dict(), save_path)
