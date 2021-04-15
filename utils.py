from functools import partial
import numpy as np
import os
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *


def cross_val_score_torch(estimator, X, y, scoring,cv = None,device='cpu'):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    score_test = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train=torch.from_numpy(X_train).to(torch.float32).to(device)
        y_train=torch.from_numpy(y_train).to(torch.float32).to(device)
        X_test=torch.from_numpy(X_test).to(torch.float32).to(device)
        y_test=torch.from_numpy(y_test).to(torch.float32).to('cpu')
        
        
        estimator.fit(X_train,y_train)
        preds=estimator.forward(X_test)
        
        if device !='cpu':
            preds=preds.detach().cpu().clone()

        score=scoring(y_test, preds)
        score_test.append(score)
        
    return score_test

