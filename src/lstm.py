import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
# import torch.nn.Parameter as Parameter

_VF = torch._C._VariableFunctions

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
device = cpu

def rectify(x):
    relu = nn.ReLU()
    return relu(x)

class LstmModule(nn.Module):
    def __init__(self, input_units, output_units, hidden_units, batch_size, bias = True, num_chunks = 4, embedding_dim = 50):
        super(LstmModule, self).__init__()

        input_size = input_units
        hidden_size = hidden_units
        # num_chunks = 2
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(batch_size, num_chunks * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(batch_size, num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(batch_size, num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(batch_size, num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input_, hx = None):
        if hx is None:
            hx = input_.new_zeros(self.batch_size, self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        hprev, cprev = hx
        hprev = hprev.view(self.batch_size, self.hidden_size, 1)
        
        w_x = self.bias_ih + torch.matmul(self.weight_ih, input_).view(self.batch_size, -1)
        w_h = self.bias_hh + torch.matmul(self.weight_hh, hprev).view(self.batch_size, -1)
        w_w = (w_x + w_h).transpose(0, 1)
        
        i = self.sigmoid(w_w[0 : self.hidden_size]).transpose(0, 1)
        f = self.sigmoid(w_w[self.hidden_size : 2*self.hidden_size]).transpose(0, 1)
        o = self.sigmoid(w_w[2*self.hidden_size : 3*self.hidden_size]).transpose(0, 1)
        g = self.tanh(w_w[3*self.hidden_size : 4*self.hidden_size]).transpose(0, 1)

        c = (f * cprev) + (i * g)
        h = o * self.relu(c)

        return (h, c), o, i, f, g, cprev

class LSTM(nn.Module):
    def __init__(self, input_units, hidden_units, vocab_size, batch_size = 1, embedding_dim = 50, output_units = 10, num_layers = 1, dropout=0):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size

        for layer in range(num_layers):
            layer_input_units = input_units if layer == 0 else hidden_units
            cell = LstmModule(input_units = input_units, output_units = output_units, hidden_units = hidden_units, batch_size = batch_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.embedding_layer = torch.nn.Embedding(vocab_size, input_units)
        self.linear = nn.Linear(hidden_units * num_layers, 2)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, max_time = 50) :
        layer_output = None
        all_layers_last_hidden = []
        state = None
        input_ = input_.long()
        max_time = len(input_)
        all_hidden, all_outputs, all_i, all_f, all_g, all_cprev = [], [], [], [], [], []

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            for time in range(max_time):
                input_emb = self.embedding_layer(input_[time]).view(self.batch_size, self.input_units, 1)
                state, outs, i, f, g, cprev = cell(input_ = input_emb, hx = state)
                h, c = state
                all_hidden.append(h.tolist())
                out = self.linear(h)
                all_outputs.append(out.tolist())
                all_i.append(i.tolist())
                all_f.append(f.tolist())
                all_g.append(g.tolist())
                all_cprev.append(cprev.tolist())
        
        hlast, clast = state
        softmax_out = self.linear(hlast)
        allh = torch.Tensor(all_hidden).transpose(0, 1)
        allo = torch.Tensor(all_outputs).transpose(0, 1)
        alli = torch.Tensor(all_i).transpose(0, 1)
        allf = torch.Tensor(all_f).transpose(0, 1)
        allg = torch.Tensor(all_g).transpose(0, 1)
        allc = torch.Tensor(all_cprev).transpose(0, 1)
        all_hidden = allh.tolist()
        all_outputs = allo.tolist()
        all_i = alli.tolist()
        all_f = allf.tolist()
        all_g = allg.tolist()
        all_cprev = allc.tolist()
        # print('softmax_out', type(softmax_out), softmax_out)
        return softmax_out, all_hidden, all_outputs, all_i, all_f, all_g, all_cprev
