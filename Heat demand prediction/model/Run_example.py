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
from train_model import train
from Prediction import predict
from pathlib import Path

def main():
    DATA_DIR = Path(__file__).parent.absolute() / 'data'

    #filename = 'Heavy_weight.txt' #Dettached_M_weight
    filename = 'Light_weight.txt'
    seq_length = 12
    # Prepare the data.
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(DATA_DIR.joinpath(filename), seq_length)
        
    input_size = 6
    hidden_size = 20
    num_layers = 1
    num_classes = 1  
    bidirectional = True
            
    # name of the save model
    MODEL_DIR = Path(__file__).parent.absolute()
    model_filename = "heat_demand.pt"
    
    # uncomment lines below to re-train the model
    num_epochs = 2500
    # learning rate
    learning_rate = 0.01
    #train(filename,seq_length,num_epochs,learning_rate,
     #            input_size,hidden_size,num_layers,num_classes,bidirectional,MODEL_DIR.joinpath(model_filename))
    
    
 
    # call prediction function.
    data_predict = predict(testX, seq_length, input_size, hidden_size,
                           num_layers, num_classes, bidirectional, MODEL_DIR.joinpath(model_filename))
    
    dataY_plot   = testY.data.numpy()    
    sc_Y=load('sc_Y.bin')
    data_predict = sc_Y.inverse_transform(data_predict)
    data_predict[data_predict < 0] = 0
    dataY_plot   = sc_Y.inverse_transform(dataY_plot)
    
    # plot the results
    
    fig, axs = plt.subplots(2,figsize=(20,12))
    axs[0].plot(dataY_plot[:,0],label='measured')
    axs[0].plot(data_predict[:,0],label = 'predict')
    axs[1].plot(dataY_plot[:,0],label='measured')
    axs[1].plot(data_predict[:,0],label = 'predict')
    axs[0].title.set_text('Zoom_in')
    axs[1].title.set_text('Heat demand')
    axs[0].set_xlim([1500,2000])
    #axs[1].set_xlim([500,1000])
    axs[0].legend()
    axs[1].legend()
    plt.show()
  

if __name__ == "__main__":
    
    main()  # temporary solution, recommended syntax
    
    