#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 06:07:32 2020

@author: arpan

@Description: Auto-Encoder model definition
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.2, bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)# only last neuron output send to decoder


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        
    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out

class EncoderGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_directions = int(bidirectional) + 1
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                          bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
#        nn.init.orthogonal_(self.gru.weight_ih_l0)
#        nn.init.xavier_uniform_(self.gru.weight_ih_l0, gain=np.sqrt(2))
#        nn.init.xavier_uniform_(self.gru.weight_hh_l0, gain=np.sqrt(2))
#        nn.init.orthogonal_(self.gru.weight_ih_l0, gain=np.sqrt(2))
#        nn.init.orthogonal_(self.gru.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.n_directions, x.size(0), \
                             self.hidden_size).to(device)
#        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate gru
        out, _ = self.gru(x, (h0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)# only last neuron output send to decoder


class DecoderGRU(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.n_directions = int(bidirectional) + 1
        self.gru = nn.GRU(hidden_size, output_size, num_layers, batch_first=True, 
                          bidirectional=bidirectional)
        self.relu = nn.ReLU()
#        nn.init.orthogonal_(self.gru.weight)

        # initialize weights
#        nn.init.xavier_uniform_(self.gru.weight_ih_l0, gain=np.sqrt(2))
#        nn.init.xavier_uniform_(self.gru.weight_hh_l0, gain=np.sqrt(2))
#        nn.init.orthogonal_(self.gru.weight_ih_l0, gain=np.sqrt(2))
#        nn.init.orthogonal_(self.gru.weight_hh_l0, gain=np.sqrt(2))
        
    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.n_directions, x.size(0), \
                             self.output_size).to(device)

        # forward propagate gru
        out, _ = self.gru(x, (h0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out
    

class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length = 6,
                 bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        self.sequence_length = sequence_length
        self.encoder = EncoderGRU(input_size, hidden_size, num_layers, bidirectional)
        self.decoder = DecoderGRU(hidden_size, input_size, num_layers, bidirectional)

    def forward(self, x):
        encoded_x = self.encoder(x).expand(-1, self.sequence_length, -1)
        decoded_x = self.decoder(encoded_x)

        return decoded_x
