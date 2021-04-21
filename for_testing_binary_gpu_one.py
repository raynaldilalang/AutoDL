import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from hyperband import HyperBandTorchSearchCV
from Models import MLPClassifier
from utils import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
# from multiprocessing.pool import Pool

def child(z):
    device='cuda'
    config={'list_hidden_layer': [16, 32, 64, 64, 32], 'input_size': 11, 'output_size': 1, 'batch_size': 256, 'epoch': 10, 'activation': 'ReLU', 'optimizer': 'Adam', 'lr': 0.013100488985051721, 'drop_rate': 0.020882414665211624, 'l1': 0.00024315820298144713, 'l2': 0.056070623526488214, 'batch_norm': True, 'device': device}

    df = pd.read_csv('for_testing_binary.csv')
    y = df.pop('target').to_frame().to_numpy()
    df.pop('id')
    X = df.copy().to_numpy()
    X = torch.from_numpy(X).to(torch.float32).to(device)
    y = torch.from_numpy(y).to(torch.float32).to(device)
    model = MLPClassifier(**config)
    model.fit(X, y)
    print(model.predict(X))
    print(model.predict_proba(X))

def parent(x):
    with MyPool(1) as p:
        p.map(child, [1, 2])

if __name__ == "__main__":
    with MyPool(1) as p:
        p.map(parent, [1, 2])
    # cross_val_score_torch(model, X, y, roc_auc_score, cv=5, n_jobs=5, device='cuda')