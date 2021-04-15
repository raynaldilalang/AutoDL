import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset,DataLoader

import pandas as pd
import numpy as np

from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self,list_hidden_layer,input_size,output_size,batch_size,lr,epoch,droprates,reguralization={'l1':0,'l2':0},device='cpu'):
        super().__init__()
        self.input_size=input_size
        self.list_layer = list_hidden_layer
        self.output_size=output_size
        
        self.epoch=epoch
        self.batch_size=batch_size
        self.lr=lr
        self.droprates=droprates
        self.reguralization=reguralization

        self.device=device
        
        hasil=[]
        hasil+=([nn.Dropout(p=droprates)])
        
        for i,num_layer in enumerate(list_hidden_layer):
            if i ==0:
                hasil+=([nn.Linear(input_size,num_layer),nn.SELU()])
            elif i == len(list_hidden_layer)-1:
                hasil+=([nn.Linear(list_hidden_layer[i-1],num_layer),nn.SELU()])
                hasil+=([nn.Linear(num_layer,output_size),nn.Sigmoid()])
            else:
                hasil+=([nn.Linear(list_hidden_layer[i-1],num_layer),nn.SELU()])
        self.layers=nn.Sequential(*hasil)
    def forward(self,x):
        return self.layers(x)
    def train(self,dataset):
        
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True)
        
        self.to(self.device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(),betas=(0.9, 0.999), lr =self.lr,weight_decay=self.reguralization['l2'])

        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(inputs)
            reg_loss = 0
            factor=0

            if self.reguralization['l1']>0:
                l1_crit = nn.L1Loss(size_average=False)
                for param in self.parameters():
                    reg_loss += l1_crit(param,target=torch.zeros_like(param))

                factor = self.reguralization['l1']

            loss = criterion(outputs, labels)
            loss += factor * reg_loss
            loss.backward()
            optimizer.step()
        self.loss = loss
                
    def fit(self,X,y):
        dataset=TensorDataset(X,y)
        for epoch in tqdm(range(self.epoch)):
            self.train(dataset)
            # if epoch % 50 ==1:
            #     print(self.loss)

