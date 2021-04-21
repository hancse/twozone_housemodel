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


def predict(pred_input,seq_length,input_size,hidden_size,
            num_layers,num_classes,bidirectional,PATH):
    
    ''' Function to predict the heat demand needed for the next hour.
    
     Args:
        pred_input:     input data for the prediction model.
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
    
    train_predict = model(pred_input)    
    data_predict = train_predict.data.numpy()
    
    return data_predict
   

if __name__ == "__main__":
    
    
    #filename = 'Heavy_weight.txt'
    filename = 'Light_weight.txt'

    seq_length = 12
    # Prepare the data.
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)

    # model parameters    
    input_size = 6
    hidden_size = 20
    num_layers = 1
    num_classes = 1
    bidirectional = True
    PATH = "heat_demand.pt"
    
    # prediction and transform the data back to it original form
    data_predict = predict(testX,seq_length,input_size,hidden_size,
                           num_layers,num_classes,bidirectional,PATH)
   
    dataY_plot   = testY.data.numpy()
    
    sc_Y=load('sc_Y.bin')
    data_predict = sc_Y.inverse_transform(data_predict)
    data_predict[data_predict < 0] = 0
    dataY_plot   = sc_Y.inverse_transform(dataY_plot)
    
    # weather data.
    sc_X=load('sc_X.bin')
    plotdataX   = testX.data.numpy()
    plotdataX   = sc_X.inverse_transform(plotdataX)
    
    # Plot the results
    
    fig, axs = plt.subplots(2,figsize=(20,12))
    axs[0].plot(plotdataX[:,0,2],label='Outdoor_Temperature')
    axs[1].plot(dataY_plot[:,0],label='measured')
    axs[1].plot(data_predict[:,0],label = 'predict')
    axs[0].title.set_text('Temperature')
    axs[1].title.set_text('Heat demand')
    #axs[0].set_xlim([500,1000])
    #axs[1].set_xlim([500,1000])
    axs[0].legend()
    axs[1].legend()
    plt.show()
    