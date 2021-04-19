# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:17:58 2021

@author: TrungNguyen
"""
import torch
from LSTM_model_struct import LSTM
from read_data import data_preprocess
import matplotlib.pyplot as plt
from joblib import load


def predict(filename,seq_length,input_size,hidden_size,
            num_layers,num_classes,bidirectional,PATH):
    
   
    model = LSTM(num_classes, input_size, hidden_size, num_layers,bidirectional,seq_length)
    model.load_state_dict(torch.load(PATH))
    
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
    
    train_predict = model(dataX)
    
    data_predict = train_predict.data.numpy()
    
    return data_predict, dataX, dataY, trainX, trainY, testX, testY
   

if __name__ == "__main__":
    
    filename = 'Heavy_weight.txt'
    seq_length = 12        
    input_size = 6
    hidden_size = 20
    num_layers = 1
    num_classes = 1
    bidirectional = True
    PATH = "heat_demand.pt"
    
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