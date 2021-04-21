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

def multi_steps_prediction(pred_input,pred_hour,seq_length,input_size,hidden_size,
                           num_layers,num_classes,bidirectional,PATH):
    
    ''' Function to predict the heat demand needed for the next hours (more than 1 hours).
    
     Args:
        pred_input:     input data for the prediction model.
        pred_hour :     number of predicted hours ahead of time
        seq_length:     the number of pass input points which needed 
                            for predicting the future value. 
        
        num_epochs:     number of times all the data has passed to the model.
        
        learning_rate:  step size at each iteration while moving 
                            toward a minimum of a loss function
      input_size:       number of input features.
      hidden_size:      number of hidden layer.
      num_classes:      number of outputs.
      bidirectional:    True or False.
      PATH:             name of the model *.pt (pytorch model)
    
    Returns:
       
        lstm:           A train lstm models with the structure define as input.   
    
    '''
    
    model = LSTM(num_classes, input_size, hidden_size, num_layers,bidirectional,seq_length)
    model.load_state_dict(torch.load(PATH))
    
    #dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
        
    # Copy a test set to a new set
    #testX.shape
    #Create an empty array
    
    predict =np.array([])
    
    for i in range(pred_hour):
        # Select a subset of 12 hour data from the test set
        indices = torch.tensor([i])
        subtest= torch.index_select(pred_input,0,indices)
        # predict the next hour
        temp    = model(subtest)
        # Replace the next value in the test set with a predicted value
        # for predicting the next time step
        pred_input[i+1,-1,0]= temp
        # Add all prediction value together. 
        temp = temp.detach().numpy()
        predict = np.append(predict,temp)
        
    
    
    return predict


if __name__ == "__main__":
    
    #filename = 'Heavy_weight.txt'
    filename = 'Light_weight.txt'
    seq_length = 12
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
        
    input_size = 6
    hidden_size = 20
    num_layers = 1
    num_classes = 1
    bidirectional = True
    PATH = "heat_demand.pt"
    
    pred_hour = 24*50    # 10 days prediction
    predict = multi_steps_prediction(testX,pred_hour,seq_length,input_size,hidden_size,
                                                      num_layers,num_classes,bidirectional,PATH)
    
    sc_Y=load('sc_Y.bin')      
    predict_tf = np.reshape(predict, (len(predict), 1))
    predict_tf = sc_Y.inverse_transform(predict_tf)
    predict_tf[predict_tf < 0] = 0
    Y_val_plot = sc_Y.inverse_transform(testY)
    
    # Plot the results
    
    fig, axs = plt.subplots(2,figsize=(20,12))
    axs[0].plot(Y_val_plot,label='measured')
    axs[0].plot(predict_tf,label='predicted')
    axs[1].plot(Y_val_plot,label='measured')
    axs[1].plot(predict_tf,label = 'predicted')
    axs[0].title.set_text('Zoom_in')
    axs[1].title.set_text('Heat demand')
    axs[0].set_xlim([0,pred_hour])
    #axs[1].set_xlim([500,1000])

    axs[0].legend()
    axs[1].legend()
    plt.show()
    
    

    
    