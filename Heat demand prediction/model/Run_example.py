# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:39:01 2021

@author: TrungNguyen
"""
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
from train_model import train
from Prediction import predict

def main():
    
    filename = 'Heavy_weight.txt'
    seq_length = 12
        
    input_size = 6
    hidden_size = 20
    num_layers = 1
    num_classes = 1  
    bidirectional = True
    
    num_epochs = 2000
    # learning rate
    learning_rate = 0.01
    
    # name of the save model 
    PATH = "heat_demand.pt"
    
    # uncomment 2 lines below to re-train the model
    
    #lstm = train(filename,seq_length,num_epochs,learning_rate,
    #             input_size,hidden_size,num_layers,num_classes,bidirectional,PATH)
    
    
 
    
    data_predict, dataX, dataY, trainX, \
        trainY, testX, testY = predict(filename,seq_length,input_size,hidden_size,
                                       num_layers,num_classes,bidirectional,PATH)
   
    dataY_plot   = dataY.data.numpy()
    
    sc_Y=load('sc_Y.bin')
    data_predict = sc_Y.inverse_transform(data_predict)
    data_predict[data_predict < 0] = 0
    dataY_plot   = sc_Y.inverse_transform(dataY_plot)
    
    plt.figure(figsize=(17,6)) #plotting
    
    plt.xlim([0,200])
    #plt.xlim([0,0])
    plt.plot(dataY_plot[:,0],label='measured')
    plt.plot(data_predict[:,0],label = 'predict')
    plt.suptitle('Time-Series Prediction')
    plt.legend()
    plt.show()
  

if __name__ == "__main__":
    
    main()  # temporary solution, recommended syntax
    
    