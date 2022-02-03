
import matplotlib.pyplot as plt
import numpy as np

from train_model import train
from load_and_preprocess_training import load_excel_data, preprocess_training
from LSTM_model_struct import LSTM
from pathlib import Path
import matplotlib
matplotlib.use("Qt5Agg")


def main():
    # set the input directory and file name
    data_dir = Path(__file__).parent.absolute() / 'data'
    # input_filename = 'test_excel.xlsx'  # this file is based on the simulink version of the house model
    input_filename = 'tst_ML.xlsx'

    # set the output directory and model filename
    model_dir = Path(__file__).parent.absolute() / 'data'  # now the same as input
    model_filename = 'test_model.pt'

    # define modeling parameters
    sequence_length = 18       # choice of 12 is based on Trung's observations, now fixed, can be determined time based
    number_input_features = 5          # now set fixed, can be obtained from the train_input tensor
    hidden_size = 20        # choice of 20 is based on Trung's observations
    number_layers = 1  # number LSTM layers, 1 layer should be sufficient for the problem, more would make the model overly complex
    number_classes = 1         # number of output classes, only heat demand
    # bidirectional = False  # is false by default, and actually is not used at this moment
    num_epochs = 1000
    learning_rate = 0.01

    # read the data
    dataset = load_excel_data(data_dir.joinpath(input_filename))

    # preprocess the training data
    train_input, train_output, test_input, test_output, input_scale, output_scale = preprocess_training(dataset,
                                                                                                        sequence_length)
    # create the network structure
    lstm_model = LSTM(number_classes, number_input_features, hidden_size, number_layers, sequence_length)

    # train the model
    trained_model = train(lstm_model, num_epochs, learning_rate, train_input, train_output,
                          model_dir.joinpath(model_filename))

    # test the model
    prediction = trained_model(test_input)  # compute prediction fro the test input

    # convert results to numpy
    prediction_np = prediction.data.numpy()
    test_out_np = test_output.data.numpy()

    # scale back to real values
    rescaled_prediction = output_scale.inverse_transform(prediction_np)
    rescaled_test_out = output_scale.inverse_transform(test_out_np)

    # compute mean squared error MSE
    mse_test = np.square(rescaled_test_out-rescaled_prediction).mean()
    print("MSE: ", mse_test)
    # plot the results

    fig, axs = plt.subplots(2, figsize=(20, 12))
    axs[0].plot(rescaled_test_out[:, 0], label='house model')
    axs[0].plot(rescaled_prediction[:, 0], label='ML predict')
    axs[1].plot(rescaled_test_out[:, 0], label='house model')
    axs[1].plot(rescaled_prediction[:, 0], label='ML predict')
    axs[0].title.set_text('Zoom_in')
    axs[1].title.set_text('Heat demand')
    # axs[0].set_xlim([7500, 8000])
    axs[0].set_xlim([1500, 1600])
    # axs[1].set_xlim([500,1000])
    axs[0].legend()
    axs[1].legend()
    plt.show()


if __name__ == "__main__":

     main()  # temporary solution, recommended syntax