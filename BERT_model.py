#!/usr/bin/env python

# Inspiration for code taken from https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b

import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchtext
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, logging
import torch.optim as optim
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm.auto import tqdm
import numpy as np

logging.set_verbosity_error()
pretty_print = lambda str, progress_bar: print(str) if progress_bar is None else progress_bar.write(str)


model_name = 'bert-base-cased'
#device = 'cuda:1'
device = 'cpu'
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok = True)
tokenizer = BertTokenizer.from_pretrained(model_name)

MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

def make_onehot(x):
    return torch.nn.functional.one_hot(torch.tensor(int(x)), num_classes=2).to(torch.float)

# Fields
label_field = Field(sequential = True,
                    tokenize = make_onehot,
                    use_vocab = False,
                    batch_first = True,
                    dtype = torch.float,
                    is_target = True)
text_field = Field(use_vocab = False,
                   tokenize = tokenizer.encode,
                   lower = False,
                   include_lengths = False,
                   batch_first = True,
                   fix_length = MAX_SEQ_LEN,
                   pad_token = PAD_INDEX,
                   unk_token = UNK_INDEX)
fields = [('comment', text_field), ('label', label_field)]

# Save and Load Functions
def save_checkpoint(save_path, model, valid_loss, progress_bar = None):
    if save_path == None:
        return    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}    
    torch.save(state_dict, save_path)
    pretty_print(f'Model saved to ==> {save_path}', progress_bar)

def load_checkpoint(load_path, model, progress_bar = None):
    if load_path==None:
        return   
    state_dict = torch.load(load_path, map_location=device)
    pretty_print(f'Model loaded from <== {load_path}', progress_bar)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list, progress_bar = None):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}  
    torch.save(state_dict, save_path)
    pretty_print(f'Model saved to ==> {save_path}', progress_bar)

def load_metrics(load_path, progress_bar = None):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    pretty_print(f'Model loaded from <== {load_path}', progress_bar)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training Function
def train(model,
          optimizer,
          train_loader,
          valid_loader,
          eval_every,
          num_epochs = 5,
          checkpoint_dir = checkpoint_dir,
          best_valid_loss = float("Inf")):

    progress_bar = tqdm(range(num_epochs * len(train_loader)))

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []


    # training loop
    model.train()
    for epoch in range(num_epochs):
        for comments, labels in train_loader:
            labels = labels.type(torch.FloatTensor).to(device)
            comments = comments.type(torch.LongTensor).to(device)
            outputs = model(input_ids=comments, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for comments, labels in valid_loader:
                        labels = labels.type(torch.FloatTensor).to(device)
                        comments = comments.type(torch.LongTensor).to(device)
                        outputs = model(input_ids=comments, labels=labels)
                        loss = outputs.loss
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                progress_bar.write('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(f'{checkpoint_dir}/model.pt', model, best_valid_loss, progress_bar)
                    save_metrics(f'{checkpoint_dir}/metrics.pt', train_loss_list, valid_loss_list, global_steps_list, progress_bar)
    

# Evaluation Function
def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for comments, labels in test_loader:
            labels = labels.type(torch.FloatTensor).to(device)
            comments = comments.type(torch.LongTensor).to(device)
            output = model(input_ids=comments, labels=labels)

            y_pred.extend(torch.argmax(output.logits, 1).tolist())
            y_true.extend(torch.argmax(labels, 1).tolist())

    return (sklearn.metrics.accuracy_score(y_true, y_pred), sklearn.metrics.balanced_accuracy_score(y_true, y_pred))

# Training with cross-validation
accs, balaccs = [], []
for i in range(10):
    print('\n\n')
    print("----------------")
    print("cross val run", i)
    print("----------------")

    # TabularDataset
    print('Loading datasets')
    traindat, validdat, testdat = TabularDataset.splits(path = 'datasets/',
                                                        train = f'train_{i}.csv',
                                                        validation = f'dev_{i}.csv',
                                                        test = f'test_{i}.csv',
                                                        format = 'CSV',
                                                        fields = fields,
                                                        skip_header = True)

    # Iterators
    train_iter = BucketIterator(traindat,
                                batch_size = 8,
                                sort_key = lambda x: len(x.comment),
                                device=device,
                                train = True,
                                sort = True,
                                sort_within_batch = True)
    valid_iter = BucketIterator(validdat,
                                batch_size = 8,
                                sort_key = lambda x: len(x.comment),
                                device = device,
                                train = True,
                                sort = True,
                                sort_within_batch = True)
    test_iter = Iterator(testdat,
                         batch_size = 8,
                         device = device,
                         train = False,
                         shuffle = False,
                         sort = False)

    # Train model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    print('Training')
    train(model = model,
          optimizer = optimizer,
          train_loader = train_iter,
          valid_loader = valid_iter,
          eval_every = len(train_iter) // 2)
    print('Finished Training!')

    # Load best model
    load_checkpoint(f'{checkpoint_dir}/model.pt', model)

    print('Beginning test set eval')
    acc, balacc = evaluate(model, test_iter)
    print("accuracy", acc)
    print("Balanced accuracy", balacc)
    accs.append(acc)
    balaccs.append(balacc)

print("Overall accuracy", np.mean(accs))
print("Overall balanced accuracy", np.mean(balaccs))
