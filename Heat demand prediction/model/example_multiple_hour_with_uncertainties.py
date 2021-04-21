# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:42:27 2021

@author: TrungNguyen
"""
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
        internal heat gain are not known. The model will predict several hours ahead of time.
    
    '''    
    
    #filename = 'Heavy_weight.txt'
    filename = 'Light_weight.txt'
    seq_length = 12
    
    # Prepare the data.
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
        
       
    #Define and load the model.
    input_size    = 6
    hidden_size   = 20
    num_layers    = 1
    num_classes   = 1  
    bidirectional = True
       
    # name of the saved model 
    PATH = "heat_demand.pt"
        
    # load the model.
    model = LSTM(num_classes, input_size, hidden_size, num_layers,bidirectional,seq_length)
    model.load_state_dict(torch.load(PATH))
    
    
    Xtsq = testX.clone().detach()
    # Construct a predicted uncertainties array and add to the predicted set
    # Assume that Solar and internal heat gain are variables with uncertainties.
    tempset = Variable(torch.Tensor(np.zeros([Xtsq.shape[0],Xtsq.shape[1],2])))
    #tempset = Variable(torch.Tensor(np.array([[0,0]])))

    # Replace Solar and internal heat gain with zero
    # (only replace the last values of the sequences)
    Xtsq[:,:,3:5]= tempset
    #Xtsq[:,-1,3:5]= tempset
    
    
    predict =np.array([])
    pred_hour = 24*50
    
    for i in range(pred_hour):
        # Select a subset of 12 hour data from the test set
        indices = torch.tensor([i])
        subtestX= torch.index_select(Xtsq,0,indices)
        # predict the next hour
       
        temp    = model(subtestX)
       
        # Replace the next hour value in the set with a predicted value
        # for predicting the next time step
        Xtsq[i+1,-1,0]= temp
        # Add all prediction value together. 
        temp = temp.detach().numpy()
        predict = np.append(predict,temp)    
    
    # Transform the data to its original form.
    sc_Y    = load('sc_Y.bin')    
    predict = sc_Y.inverse_transform(predict)
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
    axs[0].set_xlim([0,pred_hour])
    #axs[1].set_xlim([500,1000])
    axs[0].legend()
    axs[1].legend()
    plt.show()
    

if __name__ == "__main__":
    
    main()  # temporary solution, recommended syntax