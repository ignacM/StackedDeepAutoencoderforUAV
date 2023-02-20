"""
Runmodel can be called to run StackedAutoEncoder.
"""

import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from torch.autograd import Variable
from SAE import StackedAutoEncoder


def NormalizeData(data):
    """
    Normalizes data using z-score. Can be used for minmax.
    :param data:
    :return:
    """
    # Two separate scaler options, minmax and z-score
    # scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


class RMSLELoss(torch.nn.Module):
    """
    Root mean squared log error loss
    """

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


def RunModel(learning_rate, squeeze, x_train):
    """
    Runs an autoencoder.
    :param learning_rate: lower values for longer smoother training, ad
    :param squeeze: size of encoder shape (second and third autoencoder)
    :param x_train: unsupervised training data
    :return:
    """
    # Number of epochs for full-stack training stage
    epochs = 1000

    # Creating stacked autoencoder, defining optimizer and criterion for combined training phase
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedAutoEncoder(input_shape=x_train.shape[1], encode_shape=squeeze).to(device)

    # Choice between two optimizers: SGD & Adam
    # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Mean Squared Error loss criterion
    criterion = torch.nn.MSELoss()

    # normalise x_train data + convert to tensor
    train_dataset = pd.DataFrame.to_numpy(x_train)
    train_dataset, scaler = NormalizeData(train_dataset)
    x_train_tensor = torch.tensor(train_dataset.astype(float))

    print("__________________________________________________________")
    print("FITTING AUTOENCODER")

    # Independent sub-autoencoder training with high learning rate
    model(x_train_tensor.to(device).float()).clone()

    print("Training Full-Stack")
    for epoch in range(epochs + 1):

        # Training full stacked autoencoder combined
        optimizer.zero_grad()
        # Percentage of gaussian noise to be added during full stack training
        percentage = 0.05
        noise = np.random.normal(0, x_train_tensor.std(), x_train_tensor.shape) * percentage
        noised_features = (x_train_tensor + noise)

        # Model training and optimiser steps
        encoded = model.encode(noised_features.to(device).float())
        outputs = model.reconstruct(encoded)

        # Returning the value of the actual value and the predicted value and
        # Extracting a value for every epoch
        # outputs_list=outputs[epoch,0].tolist
        # x_train_list=x_train_tensor[epoch,0].tolist

        # Dynamic Weight Loss
        # Comparing the actual value to predicted
        mse = criterion(outputs.float(), Variable(x_train_tensor.to(device).float(), requires_grad=True).float())

        # Selecting constant D depending on how big the error is
        dynamic_wl = abs(mse.item()) ** 0.5

        # Manually selecting constant C value dynamic weight loss equation
        C = 0.5
        if dynamic_wl < C:
            D = dynamic_wl / 2
        else:
            D = dynamic_wl

        # Changed the loss equation to take into account dynamic weight loss
        train_loss = D * criterion(outputs.float(),
                                   Variable(x_train_tensor.to(device).float(), requires_grad=True).float())
        train_loss.backward()
        optimizer.step()
        loss = train_loss.item()

        if epoch % 1000 == 0:
            print("epoch : {}/{}, loss = {:.11f}".format(epoch, epochs, loss))

    return model, x_train_tensor, scaler

    return 0
