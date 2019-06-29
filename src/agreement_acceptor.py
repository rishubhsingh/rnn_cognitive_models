import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import six
import pickle

from lstm_model import LSTMModel
from lstm import LSTM
from utils import gen_inflect_from_vocab, dependency_fields

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
device = cpu

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

class RNNAcceptor(LSTMModel):

    def create_train_and_test(self, examples, test_size, data_name, save_data=False):
        d = [[], []]
        for i, s, dep in examples:
            d[i].append((i, s, dep))
        random.seed(1)
        random.shuffle(d[0])
        random.shuffle(d[1])
        if self.equalize_classes:
            l = min(len(d[0]), len(d[1]))
            examples = d[0][:l] + d[1][:l]
        else:
            examples = d[0] + d[1]
        random.shuffle(examples)

        Y, X, deps = zip(*examples)
        Y = np.asarray(Y)
        X = pad_sequences(X, maxlen = self.maxlen)

        n_train = int(self.prop_train * len(X))
        # self.log('ntrain', n_train, self.prop_train, len(X), self.prop_train * len(X))
        self.X_train, self.Y_train = X[:n_train], Y[:n_train]
        self.deps_train = deps[:n_train]
        if (test_size > 0) :
            self.X_test, self.Y_test = X[n_train : n_train+test_size], Y[n_train : n_train+test_size]
            self.deps_test = deps[n_train : n_train+test_size]
        else :
            self.X_test, self.Y_test = X[n_train:], Y[n_train:]
            self.deps_test = deps[n_train:]

        if (save_data) :
            with open('X_' + data_name + '_data.pkl', 'wb') as f:
                pickle.dump(X, f)
            with open('Y_' + data_name + '_data.pkl', 'wb') as f:
                pickle.dump(Y, f)
            with open('deps_' + data_name + '_data.pkl', 'wb') as f:
                pickle.dump(deps, f)

    def load_train_and_test(self, test_size, data_name):
        # Y = np.asarray(Y)
        # X = pad_sequences(X, maxlen = self.maxlen)

        with open('../grammar_data/' + data_name + '_v2i.pkl', 'rb') as f:
            self.vocab_to_ints = pickle.load(f)

        with open('../grammar_data/' + data_name + '_i2v.pkl', 'rb') as f:
            self.ints_to_vocab = pickle.load(f
)
        X = []
        Y = []

        with open('../grammar_data/X_' + data_name + '_data.pkl', 'rb') as f:
            X = pickle.load(f)

        with open('../grammar_data/Y_' + data_name + '_data.pkl', 'rb') as f:
            Y = pickle.load(f)

        with open('../grammar_data/deps_' + data_name + '_data.pkl', 'rb') as f:
            deps = pickle.load(f)

        n_train = int(self.prop_train * len(X))
        self.X_train, self.Y_train = X[:n_train], Y[:n_train]
        self.deps_train = deps[:n_train]

        if (test_size > 0) :
            self.X_test, self.Y_test = X[n_train : n_train+test_size], Y[n_train : n_train+test_size]
            self.deps_test = deps[n_train : n_train+test_size]
        else :
            self.X_test, self.Y_test = X[n_train:], Y[n_train:]
            self.deps_test = deps[n_train:]

    def create_model(self):
        self.log('Creating model')
        self.log('vocab size : ' + str(len(self.vocab_to_ints)))
        self.model = LSTM(input_units = self.maxlen ,hidden_units = self.hidden_dim, vocab_size = len(self.vocab_to_ints)+1)#.to(device)

    def results(self):
        self.log('Processing test set')
        predicted = []
        x_test = torch.tensor(self.X_test, dtype=torch.long)#.to(cpu)
        # x_test = self.X_test
        self.log(str(len(self.X_train)) + ', ' + str(len(x_test)))

        with torch.no_grad():
            for index in range(len(x_test)) :
                pred, hidden, output = self.model(x_test[index])
                if (pred[0][0] > pred[0][1]) :
                    predicted.append([0])
                else :
                    predicted.append([1])
            recs = []
            columns = ['correct', 'prediction', 'label'] + dependency_fields
            for dep, prediction in zip(self.deps_test, predicted):
                prediction = self.code_to_class[prediction[0]]
                recs.append((prediction == dep['label'], prediction, dep['label']) + tuple(dep[x] for x in dependency_fields))
        
        self.test_results = pd.DataFrame(recs, columns=columns)
        xxx = self.test_results['correct']
        self.log('Accuracy : ' + str(sum(xxx)))
        return sum(xxx)

    def results_verbose(self, df_name='_verbose_.pkl'):
        self.log('Processing test set')
        predicted, all_hidden, all_output = [], [], []
        x_test = torch.tensor(self.X_test, dtype=torch.long)
        self.log(str(len(self.X_train)) + ', ' + str(len(x_test)))

        with torch.no_grad():
            for index in range(len(x_test)) :
                if (index % 1000 == 0):
                    self.log(index)
                pred, hidden, output = self.model(x_test[index])
                # all_hidden.append(hidden)
                # all_output.append(output)
                if (pred[0][0] > pred[0][1]) :
                    predicted.append([0])
                else :
                    predicted.append([1])
            recs = []
            columns = ['correct', 'prediction', 'label'] + dependency_fields
            for dep, prediction in zip(self.deps_test, predicted):
                prediction = self.code_to_class[prediction[0]]
                recs.append((prediction == dep['label'], prediction, dep['label']) + tuple(dep[x] for x in dependency_fields))
        
        self.test_results = pd.DataFrame(recs, columns=columns)
        self.test_results.to_pickle(df_name)
        # self.test_results['activations'] = all_hidden
        # self.test_results['outputs'] = all_output
        # self.test_results.to_pickle(df_name)
        xxx = self.test_results['correct']
        self.log('Accuracy : ' + str(sum(xxx)))
        return sum(xxx)

    def results_train(self):
        self.log('Processing train set')
        predicted = []
        x_train = torch.tensor(self.X_train, dtype=torch.long)#.to(cpu)
        self.log(len(x_train))
        with torch.no_grad():
            for index in range(len(x_train)) :
                pred = self.model(x_train[index])
                if (pred[0][0] > pred[0][1]) :
                    predicted.append([0])
                else :
                    predicted.append([1])
            recs = []
            columns = ['correct', 'prediction', 'label'] + dependency_fields
            for dep, prediction in zip(self.deps_train, predicted):
                prediction = self.code_to_class[prediction[0]]
                recs.append((prediction == dep['label'], prediction, dep['label']) + tuple(dep[x] for x in dependency_fields))
        
        self.test_results = pd.DataFrame(recs, columns=columns)
        xxx = self.test_results['correct']
        self.log('Accuracy : ' + str(sum(xxx)))
        return sum(xxx)

class PredictVerbNumber(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'VBZ': 0, 'VBP': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        v = int(dep['verb_index']) - 1
        tokens = dep['sentence'].split()[:v]
        return tokens

class InflectVerb(PredictVerbNumber):
    '''
    Present all words up to _and including_ the verb, but withhold the number
    of the verb (always present it in the singular form). Supervision is
    still the original number of the verb. This task allows the system to use
    the semantics of the verb to establish the dependency with its subject, so
    may be easier. Conversely, this may mess up the embedding of the singular
    form of the verb; one solution could be to expand the vocabulary with
    number-neutral lemma forms.
    '''

    def __init__(self, *args, **kwargs):
        super(InflectVerb, self).__init__(*args, **kwargs)
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        v = int(dep['verb_index']) - 1
        tokens = dep['sentence'].split()[:v+1]
        if dep['verb_pos'] == 'VBP':
            tokens[v] = self.inflect_verb[tokens[v]]
        return tokens

class CorruptAgreement(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        tokens = dep['sentence'].split()
        if random.random() < 0.5:
            dep['label'] = 'ungrammatical'
            v = int(dep['verb_index']) - 1
            tokens[v] = self.inflect_verb[tokens[v]]
            dep['sentence'] = ' '.join(tokens)
        else:
            dep['label'] = 'grammatical'
        return tokens


class GrammaticalHalfSentence(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        tokens = dep['sentence'].split()
        v = int(dep['verb_index']) - 1
        tokens = tokens[:v+1]
        if random.random() < 0.5:
            dep['label'] = 'ungrammatical'
            tokens[v] = self.inflect_verb[tokens[v]]
        else:
            dep['label'] = 'grammatical'
        dep['sentence'] = ' '.join(tokens[:v+1])
        return tokens

class GramHalfPlusSentence(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        tokens = dep['sentence'].split()
        v = int(dep['verb_index']) - 1
        tokens = tokens[:v+1 + self.len_after_verb]
        if random.random() < 0.5:
            dep['label'] = 'ungrammatical'
            tokens[v] = self.inflect_verb[tokens[v]]
        else:
            dep['label'] = 'grammatical'
        dep['sentence'] = ' '.join(tokens[:v+1 + self.len_after_verb])
        return tokens