"""
Includes StackedAutoEncoder

"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


# Class for individual sub-denoising autoencoder
class DAE(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
        super().__init__()
        # Linear encoding layer
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_size, out_features=output_size
        )
        # Linear decoding layer
        self.decoder_output_layer = nn.Linear(
            in_features=output_size, out_features=input_size
        )
        # Loss criterion and optimizer during greedy training stages
        self.criterion = nn.MSELoss()
        # Choice between two optimizers, SGD and Adam
        # self.optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features, i, t):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Percentage of gaussian distributed noise to corrupt data
        features = features.detach().cpu()
        percentage = 0.05
        noise = np.random.normal(0, features.std(), features.shape) * percentage
        noised_features = (features+noise).to(device)

        # Inference and take steps
        encoded = self.encode(noised_features.float())
        reconstructed = self.reconstruct(encoded)
        self.optimizer.zero_grad()
        loss = self.criterion(reconstructed.float(), Variable(features.to(device).float(), requires_grad=True))

        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        if i % 1000 == 0:
            print("epoch : {}/{}, loss = {:.11f}".format(i, t, loss))

        return encoded
    
    def encode(self, features):
        # Pass through encoding layer & ReLU activation
        activation = self.encoder_hidden_layer(features)
        code = torch.relu(activation)
        return code
    
    def reconstruct(self, features):
        # Pass through decoding layer
        activation = self.decoder_output_layer(features)
        return activation


# Full stacked autoencoder class
class StackedAutoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super(StackedAutoEncoder, self).__init__()
        # 3 DAEs, all encoding into lower dimensions
        self.ae1 = DAE(kwargs["input_shape"], kwargs["encode_shape"], 1e-2)
        self.ae2 = DAE(kwargs["encode_shape"], kwargs["encode_shape"] - 2, 1e-2)
        self.ae3 = DAE(kwargs["encode_shape"] - 2, kwargs["encode_shape"] - 4, 1e-2)

    def forward(self, x):
        # Train first autoencoder for 2000 epochs
        print("Training A1")
        for i in range(0, 2000):
            a1 = self.ae1.forward(x, i, 2000)

        # Train second autoencoder for 2000 epochs
        print("Training A2")
        
        for i in range(0, 2000):
            a2 = self.ae2.forward(a1, i, 2000)

        # Train third autoencoder for 2000 epochs
        print("Training A3")
        for i in range(0, 2000):
            a3 = self.ae3.forward(a2, i, 2000)

        return self.reconstruct(a3)

    def reconstruct(self, x):
        # Fully reconstruct encoded data, x
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct

    def encode(self, features):
        # Encode input data into lower dimensionality
        a1 = self.ae1.encode(features)
        a2 = self.ae2.encode(a1)
        a3 = self.ae3.encode(a2)
       
        return a3
