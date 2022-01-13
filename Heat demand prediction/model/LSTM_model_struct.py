# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:51:46 2021

@author: TrungNguyen
"""
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
#from sklearn.preprocessing import MinMaxScaler

#seq_length = 12

class LSTM(nn.Module):
    
    
    """ Create an LSTM model structure in pytorch
    
   
    Args:
        num_classes : number of prediction outputs base on inputs (1 or 6, default value is 1)
        
        input_size : The number of expected features in the input x.

        hidden_size: The number of features in the hidden state h

        num_layers : Number of recurrent layers. E.g., setting num_layers=2 
                     would mean stacking two LSTMs together to form a stacked LSTM, 
                     with the second LSTM taking in outputs of the first LSTM and 
                     computing the final results. Default: 1

        bidirectional: If True, becomes a bidirectional LSTM. Default: False

        seq_length   : length of pass inputs data points.

       
    Returns:
       
       output of shape (seq_len, batch, num_directions * hidden_size): 
           
           tensor containing the output features (h_t) from the last layer 
           of the LSTM, for each t. If a torch.nn.utils.rnn.PackedSequence has 
           been given as the input, the output will also be a packed sequence. 
           If proj_size > 0 was specified, output shape will be 
           (seq_len, batch, num_directions * proj_size).
        
    """

    def __init__(self, num_classes, input_size, hidden_size, 
                 num_layers,bidirectional,seq_length):

        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # create a LSTM layer with input_size input features and hidden_size hidden neurons
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # connect a fully connected layer to the lstm layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        """
         h_0 of shape (num_layers * num_directions, batch, hidden_size): 
             
             tensor containing the initial hidden state for each element in the batch. 
             If the LSTM is bidirectional, num_directions should be 2, 
             else it should be 1. 
             If proj_size > 0 was specified, the shape has to be (num_layers * num_directions, batch, proj_size).

        c_0 of shape (num_layers * num_directions, batch, hidden_size): 
            
            tensor containing the initial cell state for each element in the batch.

        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
          
      
        """
        # description above states that if the LSTM is bidirectional the size of h_0 and c_0 is different, i.e.,
        # the first dimension should be multiplied by 2. This is currently not done. The bidirectionality is not used.
        # This implies that the LSTM is not bidirectional. Why is this set as True in the input?


        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        # feed the output of the lstm layer to the fully connected layer,
        # out is the final output of the forward propagation of the data.
        out = self.fc(h_out)
        
        return out
