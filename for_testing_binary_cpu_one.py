import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from hyperband import HyperBandTorchSearchCV
from Models import MLPClassifier
from utils import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

if __name__ == "__main__":
    device='cpu'
    config={'list_hidden_layer': [5, 5, 5], 'input_size': 11, 'output_size': 1, 'batch_size': 5, 'epoch': 1, 'activation': 'ReLU', 'optimizer': 'Adam', 'lr': 0.013100488985051721, 'drop_rate': 0.020882414665211624, 'l1': 0.00024315820298144713, 'l2': 0.056070623526488214, 'batch_norm': True, 'device': device}

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
    # cross_val_score_torch(model, X, y, roc_auc_score, cv=5, n_jobs=5, device='cuda')