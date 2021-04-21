# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:16:53 2021

@author: TrungNguyen
"""
from read_data import data_preprocess
from LSTM_model_struct import LSTM
import torch
#import torch.nn as nn
#from torch.autograd import Variable



# PATH = "heat_demand.pt"
def train(filename,seq_length,num_epochs,learning_rate,
          input_size,hidden_size,num_layers,num_classes,bidirectional,PATH):
    
    ''' Function to train and save the model for prediction.
    
     Args:
        
        filename:       name of the data set.
        seq_length:     the number of pass input points which needed 
                            for predicting the future value. 
        
        num_epochs:     number of times all the data has passed to the model.
        
        learning_rate:  step size at each iteration while moving 
                            toward a minimum of a loss function
      input_size:       number of input features.
      hidden_size:      number of hidden layer.
      num_classes:      number of outputs.
      bidirectional:    True or False.
      PATH:             name of the save model *.pt (pytorch model)
    
    Returns:
       
        lstm:           A train lstm models with the structure define as input.   
    
    '''
    
    
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
      
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers,bidirectional,seq_length)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    
    # Save the model for prediction.
    
    #PATH = PATH
    torch.save(lstm.state_dict(), PATH)
    
    return lstm
    
if __name__ == "__main__":
    
    # data file name
    #filename = 'Heavy_weight.txt'
    filename = 'Light_weight.txt'

    seq_length = 12
        
    # number of tranning cycle.
    num_epochs = 2000
    # learning rate
    learning_rate = 0.01
    # Train the model 
    
    input_size  = 6
    hidden_size = 20
    num_layers  = 1
    num_classes = 1
    bidirectional = True
    
    PATH = "heat_demand.pt"
    
    lstm = train(filename,seq_length,num_epochs,learning_rate,
                 input_size,hidden_size,num_layers,num_classes,bidirectional,PATH)
    print(lstm)
    
