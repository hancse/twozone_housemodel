"""
Created on Dec 22 2021

@author: MJ
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


def load_excel_data_for_model_making(filename):
    """"" read a excel file containing the data used for the machine learning
    Args:
        filename: (str) name of the file containing the data for now a test file based on the housemodel data from Trung

    Returns:
        numpy arrays for a training and test set containing the different features without the timestamps first column
        containing the heat demand which will be predicted in the model.

    """
    data_xls = pd.read_excel(filename)  # read the excel file
    data_selection = data_xls[['heat_demand', 'temperature_house', 'temperature_outside', 'setpoint_temperature',
                               'heat_solar']]  # select the columns to use for machine learning
    data_selection_np = data_selection.to_numpy()  # conversion of the data to numpy array
    training_fraction = 0.7  # fraction of the data set that will be used for training
    nr_samples = len(data_selection_np)  # total number of samples in the data set
    index_train = int(nr_samples * training_fraction)  # index for the last sample of the training set
    training_data = data_selection_np[0:index_train, :]  # training set is the first part of the time series
    test_data = data_selection_np[index_train:-1, :]  # test set is the last part of the time series

    return training_data, test_data  # numpy arrays containing the data used for training and testing


def preprocess_training(np_data, sequence_length):
    """" convert the data in the numpy array into an input format that can be used by the LSTM model
        1. normalize the data, shift with mean and scale with standard deviation the normalization parameters need to
        be saved in order to be able to use the model on new date.
        2. create a sliding window of input history for each output value. The sliding window is of of size 
        'sequence_length'
        3. split the data into training and test set. 
        4. convert the dataset to tensors suitable for torch
        
        Args:
            np_data: data set as numpy array 
        Returns:
            normalization_parameter set
            torch tensors for training and testing
    """
    # define the input and output datasets.
    input_data = np_data  # the full dataset will be used as input
    output_data = np_data[:, 0]  # the first column contains the heat demand, and is will be used to create the output

    # normalize the data
    input_scale = StandardScaler().fit(input_data)  # scaling factors for the input data
    output_scale = StandardScaler().fit(output_data.reshape(-1, 1))  # scaling factors for the output data,
    # reshape is needed to create a column input for the standard scaler

    scaled_input_data = input_scale.transform(input_data)  # scale the input data with the means and variances
    scaled_output_data = output_scale.transform(output_data.reshape(-1, 1))    # scale the output data

    # create a sliding window over the input and match the appropriate output value

    # sliding window does not work gives error:
    # AttributeError: module 'numpy.lib.stride_tricks' has no attribute 'sliding_window_view'
    # nr_input_features = np.shape(scaled_input)[1]
    # window_shape = (sequence_length, nr_input_features)
    # test_slide = np.lib.stride_tricks.sliding_window_view(input_data, window_shape, axis =0)

    # method from Trungs code:
    x = []
    y = []

    for i in range(len(scaled_input_data) - sequence_length - 1):
        _x = scaled_input_data[i:(i + sequence_length)]
        _y = scaled_output_data[i + sequence_length]
        x.append(_x)
        y.append(_y)

    x, y = np.array(x), np.array(y)

    train_in = Variable(torch.Tensor(x))
    train_out = Variable(torch.Tensor(y))

    return train_in, train_out, input_scale, output_scale


def preprocess_test(test_data, input_scale, sequence_length):
    """
    preprocess the test input data
        1. scale the data with the input scale factors obtained from the training set
        2. create the input tensor using a sliding window.
        3. create a corresponding non-scaled output array

        Note that due to the sliding window there are less outputs than samples in the data. This results in a gap
        between training and test of the size of the sequence_length


    Args:
        test_data: a numpy array with the input data for all features
        input_scale: scaling parameters for the input features

    Returns:
        test_in_tensor torch tensor that can be used with the model
        test_out a numpy array that contains the corresponding non-scaled output

    """

    scaled_test_data = input_scale.transform(test_data)  # scale the input d

    # method from Trungs code:
    test_input = []

    for i in range(len(scaled_test_data) - sequence_length - 1):
        _x = scaled_test_data[i:(i + sequence_length)]
        test_input.append(_x)

    test_input = np.array(test_input)
    test_output = test_data[sequence_length:-1, 0] # use the unscaled data to create a matching output vector

    test_in_tensor = Variable(torch.Tensor(test_input))

    return test_in_tensor, test_output


if __name__ == "__main__":

    sequence_length = 10
    excel_file = 'test_excel.xlsx'
    DATA_DIR = Path(__file__).parent.absolute() / 'data'

    train_data, test_data = load_excel_data_for_model_making (DATA_DIR.joinpath(excel_file))

    train_in_tensor, train_out_tensor, input_scale_factor, output_scale_factor = preprocess_training(train_data,
                                                                                                     sequence_length)
    test_in, test_out = preprocess_test(test_data, input_scale_factor, sequence_length)

    print(test_in.shape)
    print(test_out.shape)


