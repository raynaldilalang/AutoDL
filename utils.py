from functools import partial
import numpy as np
import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *

import multiprocessing
import torch.multiprocessing
from torch.multiprocessing import Pool

class NoDaemonProcess(torch.multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def get_score(estimator, X_train, y_train, X_test, y_test, scoring):

    device = estimator.device

    X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
    y_train = torch.from_numpy(y_train).to(torch.float32).to(device)
    X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).to(torch.float32).to('cpu')

    estimator.fit(X_train, y_train)
    y_pred = estimator.forward(X_test).detach().cpu().clone()
    score = scoring(y_test, y_pred)

    return score

def cross_val_score_torch(estimator, X, y, scoring, cv=None, n_jobs=None):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    score_test = []
    listToTrain = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        listToTrain.append((estimator, X_train, y_train, X_test, y_test, scoring))

    torch.multiprocessing.set_start_method('spawn', force=True)
    with MyPool(n_jobs) as p:
        score_test = p.starmap(get_score, listToTrain)

    return np.array(score_test)
