import json
import multiprocessing
import os
import os.path as op
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pickle

import filenames
from utils import deps_from_tsv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
device = cpu

class LSTMModel(object):

    serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
                             'X_train', 'Y_train', 'deps_train',
                             'X_test', 'Y_test', 'deps_test']

    def __init__(self, filename=None, serialization_dir=None,
                 batch_size=1, embedding_size=50, hidden_dim = 50,
                 maxlen=50, prop_train=0.9, rnn_output_size=10,
                 mode='infreq_pos', vocab_file=filenames.vocab_file,
                 equalize_classes=False, criterion=None, len_after_verb=0,
                 verbose=1, output_filename='default.txt'):
        '''
        filename: TSV file with positive examples, or None if unserializing
        criterion: dependencies that don't meet this criterion are excluded
            (set to None to keep all dependencies)
        verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
        '''
        self.filename = filename
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.prop_train = prop_train
        self.mode = mode
        self.rnn_output_size = rnn_output_size
        self.maxlen = maxlen
        self.equalize_classes = equalize_classes
        self.criterion = (lambda x: True) if criterion is None else criterion
        self.len_after_verb = len_after_verb
        self.verbose = verbose
        self.output_filename = output_filename
        # self.set_serialization_dir(serialization_dir)

    def log(self, message):
        with open('logs/' + self.output_filename, 'a') as file:
            file.write(str(message) + '\n')

    def log_grad(self, message):
        with open('logs/grad_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def pipeline(self, train = True, load = False, model = '', test_size=7000, 
                 train_size=None, model_prefix='__', epochs=20, data_name='Not', 
                 activation=False, df_name='_verbose_.pkl', load_data=False, 
                 save_data=False):
        if (load_data):
            self.load_train_and_test(test_size, data_name)
        else :
            self.log('creating data')
            examples = self.load_examples(data_name, save_data, None if train_size is None else train_size*10)
            self.create_train_and_test(examples, test_size, data_name, save_data)

        self.create_model()
        if (load) :
            self.load_model(model)
        if (train) :
            self.train(epochs, model_prefix)

        print('Data : ',  data_name)
        self.log(data_name)

        if (activation) :
            acc = self.results_verbose(df_name)
        else :
            acc = self.results()

        if (test_size == -2):
            acctrain = self.results_train()

    def load_examples(self, save_data=False, n_examples=None):
        '''
        Set n_examples to some positive integer to only load (up to) that 
        number of examples
        '''
        self.log('Loading examples')
        if self.filename is None:
            raise ValueError('Filename argument to constructor can\'t be None')

        self.vocab_to_ints = {}
        self.ints_to_vocab = {}
        examples = []
        n = 0

        deps = deps_from_tsv(self.filename, limit=n_examples)

        for dep in deps:
            tokens = dep['sentence'].split()
            if len(tokens) > self.maxlen or not self.criterion(dep):
                continue

            tokens = self.process_single_dependency(dep)
            ints = []
            for token in tokens:
                if token not in self.vocab_to_ints:
                    # zero is for pad
                    x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
                    self.ints_to_vocab[x] = token
                ints.append(self.vocab_to_ints[token])

            examples.append((self.class_to_code[dep['label']], ints, dep))
            n += 1
            if n_examples is not None and n >= n_examples:
                break

        if (save_data) :
            with open('plus5_v2i.pkl', 'wb') as f:
                pickle.dump(self.vocab_to_ints, f)
            with open('plus5_i2v.pkl', 'wb') as f:
                pickle.dump(self.ints_to_vocab, f)

        return examples

    def load_model(self, model) :
        self.model = torch.load(model)

    def train(self, n_epochs=10, model_prefix='__'):
        self.log('Training')
        if not hasattr(self, 'model'):
            self.create_model()
        
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        prev_param = list(self.model.parameters())[0].clone()
        max_acc = 0
        self.log(len(self.X_train))
        x_train = torch.tensor(self.X_train, dtype=torch.long, requires_grad=False)#.to(device)
        y_train = self.Y_train #torch.tensor(self.Y_train, requires_grad=False)#.to(device)
        self.log('cpu to gpu')
        # acc = self.results()
        print(n_epochs)

        fffstart = 0

        for epoch in range(n_epochs) :
            self.log('epoch : ' + str(epoch))
            self.log_grad('epoch : ' + str(epoch))
            for index in range(fffstart, len(x_train)) :
                # self.log(index)
                if ((index+1) % 1000 == 0) :
                    self.log(index+1)
                    if ((index+1) % 3000 == 0):
                        acc = self.results()
                        if (acc >= max_acc) :
                            model_name = model_prefix + '.pkl'
                            torch.save(self.model, model_name)
                            max_acc = acc
                
                self.model.zero_grad()
                output, hidden, out = self.model(x_train[index])
                if (y_train[index] == 0) :
                    actual = torch.autograd.Variable(torch.tensor([0]), requires_grad=False)#.to(device)
                else :
                    actual = torch.autograd.Variable(torch.tensor([1]), requires_grad=False)#.to(device)
                
                loss = loss_function(output, actual)
                loss.backward(retain_graph=True)
                optimizer.step()

                if ((index) % 10 == 0) :
                    counter = 0
                    self.log_grad('index : ' + str(index))
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # print(counter, param.shape)
                            self.log_grad(str(counter) + ' : ' + str(param.grad.norm().item()))
                            counter += 1

            fffstart = 0

            acc = self.results()
            if (acc > max_acc) :
                model_name = model_prefix + '.pkl'
                torch.save(self.model, model_name)
                max_acc = acc

            # self.results_train()