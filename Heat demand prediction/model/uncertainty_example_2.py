# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:43:59 2021

@author: TrungNguyen
"""

import torch
from LSTM_model_struct import LSTM
from read_data import data_preprocess
import matplotlib.pyplot as plt
from joblib import load
import numpy as np
from torch.autograd import Variable


def main():
    
    ''' An example for running the prediction with an assumption that Q_solar and
        internal heat gain are not known.
    
    '''    
    
    #filename = 'Heavy_weight.txt'
    filename = 'Light_weight.txt'
    seq_length = 12
    
    # Prepare the data.
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
        
    Xtsq = testX.clone().detach()
    
    # Construct a predicted uncertainties array and add to the predicted set
    # Assume that Solar and internal heat gain are variables with uncertainties.
    tempset = Variable(torch.Tensor(np.zeros([Xtsq.shape[0],Xtsq.shape[1],2])))
    #tempset = Variable(torch.Tensor(np.array([[0,0]])))

    # Replace Solar and internal heat gain with zero
    # (only replace the last values of the sequences)
    Xtsq[:,:,3:5]= tempset
    #Xtsq[:,-1,3:5]= tempset

    
    #Define and load the model.
    input_size    = 6
    hidden_size   = 20
    num_layers    = 1
    num_classes   = 1  
    bidirectional = True
       
    # name of the save model 
    PATH = "heat_demand.pt"
        
    # load the model.
    model = LSTM(num_classes, input_size, hidden_size, num_layers,bidirectional,seq_length)
    model.load_state_dict(torch.load(PATH))
    
    # call prediction function
    predict_test = model(Xtsq)
    predict_data = predict_test.data.numpy()
    
    # Transform the data to its original form.
    sc_Y    = load('sc_Y.bin')    
    predict = sc_Y.inverse_transform(predict_data)
    predict[predict < 0] = 0
    Y_val   = testY.data.numpy()
    Y_val_plot = sc_Y.inverse_transform(Y_val)
    
    # Plot the results

    fig, axs = plt.subplots(2,figsize=(20,12))
    axs[0].plot(Y_val_plot,label='measured')
    axs[0].plot(predict,label='predicted')
    axs[1].plot(Y_val_plot,label='measured')
    axs[1].plot(predict,label = 'predicted')
    axs[0].title.set_text('Zoom_in')
    axs[1].title.set_text('Heat demand')
    #plt.figure(figsize=(17,6)) #plotting
    axs[0].set_xlim([1500,2000])
    #axs[1].set_xlim([500,1000])
   
    axs[0].legend()
    axs[1].legend()
    plt.show()
    


if __name__ == "__main__":
    
    main()  # temporary solution, recommended syntax