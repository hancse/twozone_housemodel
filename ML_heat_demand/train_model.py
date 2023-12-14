# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:16:53 2021

@author: TrungNguyen
"""
import torch


def train(model_network, num_epochs, learning_rate, input_data, output_data, model_file):
    """
    Function to train and save the model for prediction.
    
     Args:
        model_network: neural network structure that will be trained
        num_epochs:     number of times all the data has passed to the model.
        learning_rate:  step size at each iteration while moving 
                            toward a minimum of a loss function
        input_data:       tensor containing the input of the training data
        output_data:    tensor containing the output of the training data
        model_file:             name of the save model *.pt (pytorch model)
    
    Returns:
       
        lstm:           A train lstm models with the structure define as input.   
    """

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model_network.parameters(), lr=learning_rate)  # optimization method Adam (most used)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate) # alternative optimization method
    
    # Train the model
    for epoch in range(num_epochs):
        outputs = model_network(input_data)  # lstm.forward(trainX) propagate all data through the network
        optimizer.zero_grad()  # reset the optimizer to zero
        
        # obtain the loss function
        loss = criterion(outputs, output_data)
        
        loss.backward()  # backward propagate the losses
        
        optimizer.step()  # do one optimization step for updating the parameters
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        if epoch == num_epochs-1:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    model_output = model_network(input_data)
    final_loss = criterion(model_output,output_data)
    print("final loss: ", final_loss.item())
    # Save the model for prediction.

    torch.save(model_network.state_dict(), model_file)  # save the obtained model
    
    return model_network

    
