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
                    hasil += ([nn.Linear(num_layer, output_size), nn.Sigmoid()])

        self.layers = nn.Sequential(*hasil)

    def get_config(self):
        config = dict(zip(
            ['list_hidden_layer', 'input_size', 'output_size', 'batch_size', 'epoch',
                'activation', 'optimizer', 'loss_function', 'lr', 'drop_rate', 'l1', 'l2', 'batch_norm', 'device'],
            [self.list_layer, self.input_size, self.output_size, self.batch_size, self.epoch, self.activation,
                self.optimizer, self.loss_function, self.lr, self.drop_rate, self.l1, self.l2, self.batch_norm, self.device]
        ))

        return config

    def forward(self, x):
        return self.layers(x)

    def train(self, dataset):
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
        dataset = TensorDataset(X, y)
        if verbose:
            # for epoch in tqdm(range(self.epoch)):
            for epoch in range(self.epoch):
                self.train(dataset)
        else:
            for epoch in range(self.epoch):
                self.train(dataset)


class MLPRegressor(nn.Module):
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
        config = dict(zip(
            ['list_hidden_layer', 'input_size', 'output_size', 'batch_size', 'epoch',
                'activation', 'optimizer', 'lr', 'drop_rate', 'l1', 'l2', 'batch_norm', 'device'],
            [self.list_layer, self.input_size, self.output_size, self.batch_size, self.epoch, self.activation,
                self.optimizer, self.lr, self.drop_rate, self.l1, self.l2, self.batch_norm, self.device]
        ))

        return config

    def forward(self, x):
        return self.layers(x)

    def train(self, dataset):
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
        dataset = TensorDataset(X, y)
        if verbose:
            for epoch in tqdm(range(self.epoch)):
                self.train(dataset)
        else:
            for epoch in range(self.epoch):
                self.train(dataset)
