import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import pandas as pd
import numpy as np

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class MLPClassifier(nn.Module):
    """
    Multi Layer Perceptron model for basic classification tasks.

    Parameters
    ----------
    list_hidden_layer : array-like.
        Represents the number of neurons in each hidden layer.

    input_size : int.
        The number of features.

    output_size : int.
        The length of the target.

    bath_size : int.
        The number of data points used in one iteration of backpropagation

    epoch : int.
        The number times that the learning algorithm will work
        through the entire training dataset

    activation : str.
        Activation function used in each hidden layer (except the final one, which
        will always use Sigmoid / Softmax)

    optimizer : str.
        Module name of the optimization algorithm (should be in torch.optim)

    loss_function : str.
        Name of the loss function to calculate between the predictions and the labels

    lr : float.
        Learning rate.

    drop_rate : float.
        Drop out rate in each hidden layer.

    l1 : float.
        Penalty rate used in the Lasso Regression regularization.

    l2 : float.
        Penalty rate used in the Ridge Regression regularization.

    random_state : int.
        Seed number for random number generators.

    batch_norm : bool.
        True if batch normalization is used.

    device : str.
        Represents the device on which a torch.Tensor is or will be allocated.
        One of 'cpu' or 'cuda'. Use ':' to specify the device number to be used,
        e.g. 'cuda:1'
    """
    def __init__(self, list_hidden_layer, input_size, output_size, batch_size, epoch,
                 activation='ReLU', optimizer='Adam', loss_function='BCELoss', lr=1e-3,
                 drop_rate=0, l1=0, l2=0, random_state=None,
                 batch_norm=False, device='cpu'):
        super().__init__()

        if type(random_state)==int:
            torch.manual_seed(random_state)

        self.input_size = input_size
        self.list_layer = list_hidden_layer
        self.output_size = output_size

        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.drop_rate = drop_rate
        self.l1 = l1
        self.l2 = l2
        self.batch_norm = batch_norm

        self.activation = activation
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.device = device

        activation = eval(f'nn.{activation}')
        final_activation = nn.Sigmoid if output_size == 1 else nn.Softmax

        hasil = []
        if batch_norm:
            hasil += ([nn.BatchNorm1d(input_size)])

        for i, num_layer in enumerate(list_hidden_layer):
            if 0 < i < len(list_hidden_layer)-1:
                hasil += ([nn.Linear(list_hidden_layer[i-1],
                          num_layer), activation()])
                hasil += ([nn.Dropout(p=drop_rate)])
            else:
                if i == 0:
                    hasil += ([nn.Linear(input_size, num_layer), activation()])
                if i == len(list_hidden_layer)-1:
                    hasil += ([nn.Linear(list_hidden_layer[i-1],
                                         num_layer), activation()])
                    hasil += ([nn.Linear(num_layer, output_size), final_activation()])

        self.layers = nn.Sequential(*hasil)

    def get_config(self):
        """
        A method that returns the hyperparaneters used in the neural network.

        Returns
        -------
        config : dict.
            A dictionary of hyperpartameters used in the neural network.
        """
        config = dict(zip(
            ['list_hidden_layer', 'input_size', 'output_size', 'batch_size', 'epoch',
                'activation', 'optimizer', 'loss_function', 'lr', 'drop_rate', 'l1', 'l2', 'batch_norm', 'device'],
            [self.list_layer, self.input_size, self.output_size, self.batch_size, self.epoch, self.activation,
                self.optimizer, self.loss_function, self.lr, self.drop_rate, self.l1, self.l2, self.batch_norm, self.device]
        ))

        return config

    def forward(self, x):
        """
        A method to feed forward a batch of data points.

        Parameters
        ----------
        x : torch.Tensor.
            A batch of data points feeded through the neural network.

        Returns
        -------
        y_pred : torch.Tensor.
            Outputs / prediction results of the neural network
        """
        y_pred = self.layers(x)

        return y_pred

    def train(self, dataset):
        """
        A method to train an entire dataset on one epoch.

        Parameters
        ----------
        dataset : torch.utils.data.TensorDataset.
            TensorDataset used for training.
        """
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True)

        self.to(self.device)

        criterion = eval(f'torch.nn.{self.loss_function}()')
        optimizer = eval(
            f'torch.optim.{self.optimizer}(self.parameters(), lr=self.lr, weight_decay=self.l2)')

        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(inputs)
            reg_loss = 0
            factor = 0

            if self.l1 > 0:
                l1_crit = nn.L1Loss(size_average=False)
                for param in self.parameters():
                    reg_loss += l1_crit(param, target=torch.zeros_like(param))

                factor = self.l1

            loss = criterion(outputs, labels)

            loss += factor * reg_loss
            loss.backward()
            optimizer.step()
        self.loss = loss

    def fit(self, X, y, verbose=0):
        """
        A method to fit the neural network on a set of training dataset.

        Parameters
        ----------
        X : array-like.
            Array of predictors.

        y : array-like.
            Array of target/ labels.
        """
        dataset = TensorDataset(X, y)
        if verbose:
            # for epoch in tqdm(range(self.epoch)):
            for epoch in range(self.epoch):
                self.train(dataset)
        else:
            for epoch in range(self.epoch):
                self.train(dataset)

    def predict(self, X, threshold=0.5):
        """
        Using the trained neural network to predict the labels of some data points.

        Parameters
        ----------
        X : array-like.
            A batch of data points to predict.
        
        threshold : float.
            Prediction threshold.

        Returns
        -------
        y_pred : torch.Tensor
            Outputs / prediction results of the neural network
        """
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).to(torch.float32).to(self.device)
        
        y_pred = self.layers(X)
        y_pred = torch.where(y_pred > threshold, 1, 0)

        return y_pred

    def predict_proba(self, X):
        """
        Using the trained neural network to predict the labels of some data points.

        Parameters
        ----------
        X : array-like.
            A batch of data points to predict.
        
        threshold : float.
            Prediction threshold.

        Returns
        -------
        y_pred_proba : torch.Tensor
            Outputs / prediction results of the neural network as probabilities
        """
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).to(torch.float32).to(self.device)

        y_pred = self.layers(X)
        
        if self.output_size == 1:
            y_pred_0 = 1 - y_pred
            y_pred = torch.hstack([y_pred_0, y_pred])

        return y_pred

class MLPRegressor(nn.Module):
    """
    Multi Layer Perceptron model for basic regression tasks.

    Parameters
    ----------
    list_hidden_layer : array-like.
        Represents the number of neurons in each hidden layer.

    input_size : int.
        The number of features.

    output_size : int.
        The length of the target.

    bath_size : int.
        The number of data points used in one iteration of backpropagation

    epoch : int.
        The number times that the learning algorithm will work
        through the entire training dataset

    activation : str.
        Activation function used in each hidden layer (except the final one)

    optimizer : str.
        Module name of the optimization algorithm (should be in torch.optim)

    loss_function : str.
        Name of the loss function to calculate between the predictions and the labels

    lr : float.
        Learning rate.

    drop_rate : float.
        Drop out rate in each hidden layer.

    l1 : float.
        Penalty rate used in the Lasso Regression regularization.

    l2 : float.
        Penalty rate used in the Ridge Regression regularization.

    random_state : int.
        Seed number for random number generators.

    batch_norm : bool.
        True if batch normalization is used.

    device : str.
        Represents the device on which a torch.Tensor is or will be allocated.
        One of 'cpu' or 'cuda'. Use ':' to specify the device number to be used,
        e.g. 'cuda:1'
    """
    def __init__(self, list_hidden_layer, input_size, output_size, batch_size, epoch,
                 activation='ReLU', optimizer='Adam', loss_function='L1Loss', lr=1e-3, drop_rate=0, l1=0, l2=0,
                 batch_norm=False, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.list_layer = list_hidden_layer
        self.output_size = output_size

        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.drop_rate = drop_rate
        self.l1 = l1
        self.l2 = l2
        self.batch_norm = batch_norm

        self.activation = activation
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.device = device

        activation = eval(f'nn.{activation}')

        hasil = []
        if batch_norm:
            hasil += ([nn.BatchNorm1d(input_size)])

        for i, num_layer in enumerate(list_hidden_layer):
            if 0 < i < len(list_hidden_layer)-1:
                hasil += ([nn.Linear(list_hidden_layer[i-1],
                          num_layer), activation()])
                hasil += ([nn.Dropout(p=drop_rate)])
            else:
                if i == 0:
                    hasil += ([nn.Linear(input_size, num_layer), activation()])
                if i == len(list_hidden_layer)-1:
                    hasil += ([nn.Linear(list_hidden_layer[i-1],
                                         num_layer), activation()])
                    hasil += ([nn.Linear(num_layer, output_size)])

        self.layers = nn.Sequential(*hasil)

    def get_config(self):
        """
        A method that returns the hyperparaneters used in the neural network.

        Returns
        -------
        config : dict.
            A dictionary of hyperpartameters used in the neural network.
        """
        config = dict(zip(
            ['list_hidden_layer', 'input_size', 'output_size', 'batch_size', 'epoch', 'activation',
                'optimizer', 'loss_function', 'lr', 'drop_rate', 'l1', 'l2', 'batch_norm', 'device'],
            [self.list_layer, self.input_size, self.output_size, self.batch_size, self.epoch, self.activation,
                self.optimizer, self.loss_function, self.lr, self.drop_rate, self.l1, self.l2, self.batch_norm, self.device]
        ))

        return config

    def forward(self, x):
        """
        A method to feed forward a batch of data points.

        Parameters
        ----------
        x : torch.Tensor.
            A batch of data points feeded through the neural network.

        Returns
        -------
        y_pred : torch.Tensor.
            Outputs / prediction results of the neural network
        """
        y_pred = self.layers(x)

        return y_pred

    def train(self, dataset):
        """
        A method to train an entire dataset on one epoch.

        Parameters
        ----------
        dataset : torch.utils.data.TensorDataset.
            TensorDataset used for training.
        """
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True)

        self.to(self.device)

        criterion = eval(f'torch.nn.{self.loss_function}()')
        optimizer = eval(
            f'torch.optim.{self.optimizer}(self.parameters(), lr=self.lr, weight_decay=self.l2)')

        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(inputs)
            reg_loss = 0
            factor = 0

            if self.l1 > 0:
                l1_crit = nn.L1Loss(size_average=False)
                for param in self.parameters():
                    reg_loss += l1_crit(param, target=torch.zeros_like(param))

                factor = self.l1

            loss = criterion(outputs, labels)

            loss += factor * reg_loss
            loss.backward()
            optimizer.step()
        self.loss = loss

    def fit(self, X, y, verbose=0):
        """
        A method to fit the neural network on a set of training dataset.

        Parameters
        ----------
        X : array-like.
            Array of predictors.

        y : array-like.
            Array of target/ labels.
        """
        dataset = TensorDataset(X, y)
        if verbose:
            for epoch in tqdm(range(self.epoch)):
                self.train(dataset)
        else:
            for epoch in range(self.epoch):
                self.train(dataset)

    def predict(self, X):
        """
        Using the trained neural network to predict the target of some data points.

        Parameters
        ----------
        X : array-like.
            A batch of data points to predict.

        Returns
        -------
        y_pred : torch.Tensor
            Outputs / prediction results of the neural network
        """
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).to(torch.float32).to(self.device)
        
        y_pred = self.layers(X)

        return y_pred
