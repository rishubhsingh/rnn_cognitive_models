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
    # return x

class LstmModule(nn.Module):
    def __init__(self, input_units, output_units, hidden_units, batch_size, bias = True, num_chunks = 2, embedding_dim = 50):
        super(LstmModule, self).__init__()

        input_size = input_units
        hidden_size = hidden_units
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.igate = nn.Parameter(torch.tensor(0.5))
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.weight_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_r', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input_, hx = None):
        """
        An Elman RNN cell with tanh or ReLU non-linearity.
        h' = tanh/relu(w_{ih} x + b_{ih}  +  w_{hh} h + b_{hh})
        """
        # print(self.d_rec)

        if hx is None:
            hx = input_.new_zeros(self.hidden_size, requires_grad=False)

        w_x = torch.addmv(self.bias_ih, self.weight_ih, input_)
        w_h = torch.addmv(self.bias_hh, self.weight_hh, hx)
        w_w = (w_x + w_h)

        z = self.sigmoid(w_w[0 : self.hidden_size])
        r = self.sigmoid(w_w[self.hidden_size : 2*self.hidden_size])

        inp = torch.mv(self.weight_ir, input_)
        prevh = torch.mv(self.weight_hr, (r * hx))

        h = ((1 - z) * hx) + (z * self.relu(inp + prevh + self.bias_r))

        return h, z, r

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
        # self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_units * num_layers, 2)
        # self.softmax = nn.Softmax(dim=0)
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
        max_time = len(input_)
        all_hidden, all_outputs, all_z, all_r = [], [], [], []

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            for time in range(max_time):
                # print(time)
                # inputx = torch.tensor(input_[time], requires_grad = False)#.to(device)
                state, z_gate, r_gate = cell(input_ = self.embedding_layer(input_[time]), hx = state)
                all_hidden.append(state.tolist())
                out = self.linear(state)
                all_outputs.append(out.tolist())
                all_z.append(z_gate.tolist())
                all_r.append(r_gate.tolist())
        
        hlast = state
        softmax_out = self.linear(hlast)
        softmax_out = torch.stack([softmax_out], 0)#.to(cpu)
        return softmax_out, all_hidden, all_outputs