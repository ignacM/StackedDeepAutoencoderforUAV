#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:15:07 2020

@author: victoria
"""

import torch
import numpy as np
from SAE import StackedAutoEncoder
import pandas as pd
from sklearn import preprocessing
from torch.autograd import Variable

#min-max normalisation
def NormalizeData(data):
    
    #two separate scaler options, minmax and z-score (aka standard scaler)
    # scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
 
    return data, scaler

#root mean squared log error loss
class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def RunModel(learning_rate, squeeze, x_train):
  
    #number of epochs for full-stack training stage
    epochs = 1000

    #create stacked autoencoder, define optimizer + criterion for combined training phase
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedAutoEncoder(input_shape=x_train.shape[1],encode_shape = squeeze).to(device)
    
    #choice between two optimizers, SGD and Adam
    # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    #mse loss criterion
    criterion = torch.nn.MSELoss()
      
    #normalise x_train data + convert to tensor
    train_dataset = pd.DataFrame.to_numpy(x_train)
    train_dataset, scaler =  NormalizeData(train_dataset)
    x_train_tensor = torch.tensor(train_dataset.astype(float))
       
    print("__________________________________________________________")
    print("FITTING AUTOENCODER")

    #independent sub-autoencoder training with high learning rate
    model(x_train_tensor.to(device).float()).clone()
    
    print("Training Full-Stack")
    for epoch in range(epochs+1):
        
        #x_train full stacked autoencoder combined
        optimizer.zero_grad()     
        #precentage of guassian noise to be added during full stack training
        percentage = 0.05
        noise = np.random.normal(0,  x_train_tensor.std(),  x_train_tensor.shape) * percentage
        noised_features = ( x_train_tensor+noise)
    
        #model training + optimiser steps
        encoded = model.encode( noised_features.to(device).float())
        outputs = model.reconstruct(encoded)
        train_loss = criterion(outputs.float(), Variable(x_train_tensor.to(device).float(), requires_grad=True).float())      
        train_loss.backward()
        optimizer.step()
        loss = train_loss.item()
        
        if epoch % 1000 == 0:
            print("epoch : {}/{}, loss = {:.11f}".format(epoch, epochs, loss))

    return model, x_train_tensor, scaler
       
    return 0
        