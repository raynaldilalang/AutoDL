from functools import partial
import numpy as np
import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *

import torch.multiprocessing as mp
from torch.multiprocessing import Pool

def get_score(estimator, X_train, y_train, X_test, y_test, scoring, device):

    X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
    y_train = torch.from_numpy(y_train).to(torch.float32).to(device)
    X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).to(torch.float32).to('cpu')

    estimator.fit(X_train, y_train)
    y_pred = estimator.forward(X_test).detach().cpu().clone()
    score = scoring(y_test, y_pred)

    return score

def cross_val_score_torch(estimator, X, y, scoring, cv=None, n_jobs=None, device='cpu'):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    score_test = []
    listToTrain = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        listToTrain.append((estimator, X_train, y_train, X_test, y_test, scoring, device))

    print(f'Starting multiprocess ({n_jobs})')
    mp.set_start_method('spawn', force=True)
    with Pool(n_jobs) as p:
        score_test = p.starmap(get_score, listToTrain)

    return score_test
