# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:38:39 2021

@author: TrungNguyen
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:47:55 2021

@author: TrungNguyen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from pathlib import Path
from joblib import dump
#seq_length = 12

def data_preprocess(filename, seq_length):
    
    """ Prepare data for the model with the following steps:
    
    - Read data from the data file.
    - Do normalize the data, and save the scaler for the prediction. 
    - Split the data into traning set (80%) and test set (20%).
    - Create a time lag for training set and prediction set.
    
        https://datascience.stackexchange.com/questions/72480/what-is-lag-in-time-series-forecasting#:~:text=Lag%20features%20are%20target%20values,models%20some%20kind%20of%20momentum.
    -
   
    Args:
        
        filename:     name of the data set.
        seq_length:   the number of pass input points which needed 
                      for predicting the future value.                     
      
    Returns:
       
        dataX:       input data (train + test).    
        dataY:       prediction data.   
        trainX:      train input data.
        trainY:      train prediction data.
        testX:       test input data. 
        TestY:       test prediction data.
        
    """
    
    data_dir = Path.cwd() / 'data'    
    file = filename
  
    df = pd.read_csv(data_dir/file,index_col ='Time')
    #c_data=data[['Q']]   
    df = df[['Q','T','Ta','SP']] #SP
    #df.head(10)
    
    # Define traning inputs and prediction value.
    training_set_X = df.iloc[:,0:4].values
    training_set_Y = df.iloc[:,0:1].values
    
    # Normalize the data.
    #sc_X = MinMaxScaler(feature_range=(0,1))
    #sc_Y = MinMaxScaler(feature_range=(0,1))
    
    scaler_X = StandardScaler().fit(training_set_X)
    scaler_Y = StandardScaler().fit(training_set_Y)

    training_data_X = scaler_X.transform(training_set_X)
    training_data_Y = scaler_Y.transform(training_set_Y)
    # save Normalize parameters
    dump(scaler_X, 'sc_X.bin', compress=True)
    dump(scaler_Y, 'sc_Y.bin', compress=True)

    
    # Make a time lag between tranning and prediction.
    x = []
    y = []

    for i in range(len(training_data_X)-seq_length-1):
        _x = training_data_X[i:(i+seq_length)]
        _y = training_data_X[i+seq_length]
        x.append(_x)
        y.append(_y)
    
    x, y = np.array(x) ,np.array(y)
    # comment the y to have 6 output.
    y = np.reshape(y[:,0,], (len(y), 1))
    
    train_size = int(len(y) - (0.3*len(y)))
    #test_size  = len(y) - train_size
    
    # Convert numpy array to Torch tensors.
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))
    
    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    
    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    
    return dataX, dataY, trainX, trainY, testX, testY

if __name__ == "__main__":
    
    filename = 'Heavy_weight.txt'
    seq_length = 12

    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename,seq_length)
    
    plt.plot(dataY)

