# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:55:58 2021

@author: TrungNguyen
"""

import torch
from LSTM_model_struct import LSTM
from read_data import data_preprocess
import matplotlib.pyplot as plt
from joblib import load
import numpy as np

def multi_steps_prediction(filename,seq_length,input_size,hidden_size,
                           num_layers,num_classes,bidirectional,PATH):
    
    
    model = LSTM(num_classes, input_size, hidden_size, num_layers,bidirectional,seq_length)
    model.load_state_dict(torch.load(PATH))
    
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
        
    # Copy a test set to a new set
    #testX.shape
    #Create an empty array
    
    predict =np.array([])
    
    for i in range(len(testX)-1):
        # Select a subset of 12 hour data from the test set
        indices = torch.tensor([i])
        subtestX= torch.index_select(testX,0,indices)
        # predict the next hour
        temp    = model(subtestX)
        # Replace the next value in the test set with a predicted value
        # for predicting the next time step
        testX[i+1,-1,0]= temp
        # Add all prediction value together. 
        temp = temp.detach().numpy()
        predict = np.append(predict,temp)
        
    
    
    return predict, dataX, dataY, trainX, trainY, testX, testY


if __name__ == "__main__":
    
    filename = 'Heavy_weight.txt'
    seq_length = 12        
    input_size = 6
    hidden_size = 20
    num_layers = 1
    num_classes = 1
    bidirectional = True
    PATH = "heat_demand.pt"
    
    predict, dataX, dataY, trainX, \
        trainY, testX, testY = multi_steps_prediction(filename,seq_length,input_size,hidden_size,
                                                      num_layers,num_classes,bidirectional,PATH)
    
    sc_Y=load('sc_Y.bin')      
    predict_tf = np.reshape(predict, (len(predict), 1))
    predict_tf = sc_Y.inverse_transform(predict_tf)
    predict_tf[predict_tf < 0] = 0
    Y_val_plot = sc_Y.inverse_transform(testY)
    
    plt.figure(figsize=(17,6)) #plotting
    plt.plot(predict_tf,label='predicted')
    plt.plot(Y_val_plot,label='Mesaured')
    plt.xlim([0,1000])
    plt.legend()
    plt.show
    

    
    