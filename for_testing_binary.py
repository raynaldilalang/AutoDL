# import sys
# sys.path.append('../')

from hyperband import HyperBandTorchSearchCV
from Models import MLPClassifier
from utils import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import time

if __name__ == "__main__":
    start = time.time()

    dict_params = {
        'MLPClassifier': {
            'num_hidden_layer': (2, 6),
            'num_neuron': (2, 32, 'loguniform'),
            'input_size': [11],
            'output_size': [1],
            'batch_size': (2, 128, 'loguniform'),
            'activation': ['Tanh', 'ReLU', 'SELU','CELU', 'GELU', 'PReLU', 'SiLU'],
            'optimizer': ['Adam', 'SGD', 'Adadelta', 'RMSprop'],
            'loss_function': ['BCELoss'],
            'lr': (1e-4, 1e-1, 'loguniform'),
            'drop_rate': (1e-6, 1e-1, 'loguniform'),
            'l1': (1e-6, 1e-1, 'loguniform'),
            'l2': (1e-6, 1e-1, 'loguniform'),
            'batch_norm': [True, False],
        }
    }

    opt = HyperBandTorchSearchCV(
        estimator='MLPClassifier',
        search_spaces=dict_params['MLPClassifier'],
        max_epochs=64,
        factor=4,
        scoring=roc_auc_score,
        cv=3,
        random_state=420,
        n_jobs=1,
        device='cuda',
        gpu_ids=[0, 1, 2],
        greater_is_better=True
    )

    df = pd.read_csv('train_binary_sampled.csv')
    y = df.pop('target').to_frame().to_numpy()
    df.pop('id')
    X = df.copy().to_numpy()

    opt.fit(X, y)
    opt.best_config.to_csv('df_hasil_classifier.csv', index=False)

    end = time.time()

    print(f'\nHyperband Search took {round(end-start, 3)} seconds')
