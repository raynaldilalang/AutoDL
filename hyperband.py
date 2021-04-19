import numpy as np
import pandas as pd
import random
import math
from itertools import product
from scipy.stats import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from numpy.random import default_rng
from utils import *
import time
from tqdm import tqdm

import multiprocessing
import torch.multiprocessing
from torch.multiprocessing import Pool
from torch.multiprocessing import Process

from Models import MLP


class HyperBandTorchSearchCV:
    def __init__(self, estimator, search_spaces,
                 scoring, max_epochs, factor=3,
                 cv=1, random_state=420,
                 n_jobs=1, device='cpu', gpu_ids=None):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.scoring = scoring
        self.max_epochs = max_epochs
        self.factor = factor
        self.cv = cv
        self.random_state = random_state
        self.max_rounds = math.floor(math.log(max_epochs, factor))
        self.total_epochs = (self.max_rounds + 1) * max_epochs

        self.n_jobs = n_jobs
        self.device = device

        available_cudas = torch.cuda.device_count()
        if available_cudas == 0 and device == 'cuda':
            self.device = 'cpu'
            print(f'WARNING: No cuda devices are found, using cpu instead')
        elif available_cudas < len(gpu_ids):
            n_gpu = available_cudas
            print(f'WARNING: Only {available_cudas} cuda devices are found')
        else:
            n_gpu = len(gpu_ids)

        self.gpu_ids = gpu_ids
        if n_gpu > 0:
            self.n_device = n_gpu
        else:
            self.n_device = 1

        print(
            f'Initializing Torch HyperBand Search using {self.n_device} {self.device} devices')

    @staticmethod
    def get_cv_score(model, X, y, scoring, cv, n_jobs, verbose):
        return cross_val_score_torch(model, X, y, scoring, cv, n_jobs, verbose).mean()

    @staticmethod
    def create_cat_combinations(dict_hparam):
        cat_hparam = {}
        num_hparam = {}
        layers_hparam = {}

        for hparam in dict_hparam.keys():
            if hparam in ['num_hidden_layer', 'num_neuron']:
                layers_hparam[hparam] = dict_hparam[hparam]
            elif type(dict_hparam[hparam]) == list:
                cat_hparam[hparam] = dict_hparam[hparam]
            else:
                num_hparam[hparam] = dict_hparam[hparam]

        combinations_product = product(
            *(cat_hparam[hparam] for hparam in cat_hparam.keys()))
        combinations = pd.DataFrame(
            combinations_product, columns=cat_hparam.keys())

        return cat_hparam, num_hparam, layers_hparam, combinations

    @staticmethod
    def get_hyperparameter_configuration(cat_hparam, num_hparam, layers_hparam, combinations, n, random_state=420):
        np.random.seed(seed=random_state)
        configuration = dict.fromkeys(range(n))
        for ind in range(n):
            configuration[ind] = {'hparams': None}
            configuration[ind]['hparams'] = dict.fromkeys(
                [*cat_hparam.keys(), *num_hparam.keys(), 'list_hidden_layer']
            )
            if len(layers_hparam['num_hidden_layer']) == 3:
                try:
                    distribution = eval(
                        layers_hparam['num_hidden_layer'][2].replace("-", ""))
                    num_hidden_layer = int(distribution.rvs(
                        layers_hparam['num_hidden_layer'][0], layers_hparam['num_hidden_layer'][1]-layers_hparam['num_hidden_layer'][0]))
                except NameError:
                    print(
                        f'WARNING: Distribution {layers_hparam["num_hidden_layer"][2]} not found, generating random number uniformly.')
                    num_hidden_layer = randint.rvs(
                        layers_hparam['num_hidden_layer'][0], layers_hparam['num_hidden_layer'][1]+1)
            else:
                num_hidden_layer = randint.rvs(
                    layers_hparam['num_hidden_layer'][0], layers_hparam['num_hidden_layer'][1]+1)

            if len(layers_hparam['num_neuron']) == 3:
                try:
                    distribution = eval(
                        layers_hparam['num_neuron'][2].replace("-", ""))
                    configuration[ind]['hparams']['list_hidden_layer'] = distribution.rvs(
                        layers_hparam['num_neuron'][0], layers_hparam['num_neuron'][1]-layers_hparam['num_neuron'][0], size=num_hidden_layer).astype(int).tolist()
                except NameError:
                    print(
                        f'WARNING: Distribution {layers_hparam["num_neuron"][2]} not found, generating random number uniformly.')
                    configuration[ind]['hparams']['list_hidden_layer'] = randint.rvs(
                        layers_hparam['num_neuron'][0], layers_hparam['num_neuron'][1]+1, size=num_hidden_layer).tolist()
            else:
                configuration[ind]['hparams']['list_hidden_layer'] = randint.rvs(
                    layers_hparam['num_neuron'][0], layers_hparam['num_neuron'][1]+1, size=num_hidden_layer).tolist()

            if len(cat_hparam):
                cat_combination_num = random.randint(
                    0, len(combinations)-1)
                for hparam in cat_hparam.keys():
                    configuration[ind]['hparams'][hparam] = combinations.loc[cat_combination_num, hparam]

            if len(num_hparam):
                for hparam in num_hparam.keys():
                    if len(num_hparam[hparam]) == 3:
                        try:
                            distribution = eval(
                                num_hparam[hparam][2].replace("-", ""))
                            if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                                configuration[ind]['hparams'][hparam] = int(distribution.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0]))
                            else:
                                configuration[ind]['hparams'][hparam] = distribution.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])
                        except NameError:
                            print(
                                f'WARNING: Distribution {num_hparam[hparam][2]} not found, generating random number uniformly.')
                            if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                                configuration[ind]['hparams'][hparam] = randint.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]+1)
                            else:
                                configuration[ind]['hparams'][hparam] = uniform.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])
                    else:
                        if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                            configuration[ind]['hparams'][hparam] = randint.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]+1)
                        else:
                            configuration[ind]['hparams'][hparam] = uniform.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])

            configuration[ind]['isTrained'] = False

        return configuration

    def create_brackets(self):
        brackets = dict.fromkeys(range(self.max_rounds + 1))
        for bracket_num in range(self.max_rounds, -1, -1):
            n = math.ceil(
                (self.total_epochs/self.max_epochs) *
                (self.factor**bracket_num/(bracket_num+1))
            )
            r = self.max_epochs * self.factor**(-bracket_num)
            brackets[bracket_num] = dict.fromkeys(range(bracket_num + 1))
            for i in range(bracket_num + 1):
                brackets[bracket_num][i] = dict.fromkeys(
                    ['ni', 'ri', 'contenders'])
                brackets[bracket_num][i]['ni'] = math.floor(
                    n * self.factor**(-i))
                brackets[bracket_num][i]['ri'] = float(r * self.factor**i)
                brackets[bracket_num][i]['contenders'] = dict.fromkeys(
                    range(brackets[bracket_num][i]['ni']))

            bracket_df = pd.DataFrame.from_dict(
                brackets[bracket_num], orient='index')[['ni', 'ri']]
            print(f'Bracket {bracket_num} setup:')
            print(bracket_df.rename(
                columns={'ni': 'Number of Configurations', 'ri': 'Resource'}), '\n')

        self.brackets = brackets

    @staticmethod
    def create_model(model_name, random_state, epoch, device, **hparams):
        model = eval(f'{model_name}')(
            **hparams, epoch=int(epoch), device=device)

        return model

    @staticmethod
    def get_top_k(bracket_round, k):
        configurations = pd.DataFrame.from_dict(bracket_round, orient='index')
        configurations = configurations.sort_values(
            ['score'], ascending=False).reset_index(drop=True).head(k)
        configurations = configurations.to_dict(orient='index')

        return configurations

    def fit_multiple(self, X, y, configurations, bracket_num):
        device_used = self.device
        if device_used == 'cuda':
            device_used += f':{self.gpu_ids[bracket_num % self.n_device]}'
        list_toTrain_model = []

        for i in tqdm(range(bracket_num + 1), desc=f'Bracket {bracket_num}', position=(self.max_rounds-bracket_num)):
            for contender in range(self.brackets[bracket_num][i]['ni']):
                self.brackets[bracket_num][i]['contenders'][contender] = dict.fromkeys([
                                                                                       'hparams', 'score'])
                self.brackets[bracket_num][i]['contenders'][contender]['hparams'] = configurations[contender]['hparams']
                model = self.create_model(
                    self.estimator,
                    random_state=self.random_state,
                    epoch=self.brackets[bracket_num][i]['ri'],
                    device=device_used,
                    **configurations[contender]['hparams']
                )
                verbose = int(i == bracket_num)
                list_toTrain_model.append(
                    (model, X, y, self.scoring, self.cv, self.cv, verbose))

            torch.multiprocessing.set_start_method('spawn', force=True)
            with MyPool(self.n_jobs) as p:
                list_toTrain_score = p.starmap(
                    self.get_cv_score, list_toTrain_model)

            for contender in range(self.brackets[bracket_num][i]['ni']):
                self.brackets[bracket_num][i]['contenders'][contender]['score'] = list_toTrain_score[contender]

            configurations = self.get_top_k(
                self.brackets[bracket_num][i]['contenders'],
                k=max(math.floor(
                    self.brackets[bracket_num][i]['ni']/self.factor), 1)
            )

        configurations = configurations[0]
        end = time.time()
        # print(f'Best of Bracket {bracket_num}:', configurations)
        return configurations

    def fit(self, X, y):
        print(f'HyperBand on {self.estimator} \n')
        self.create_brackets()
        cat_hparam, num_hparam, layers_hparam, combinations = self.create_cat_combinations(
            self.search_spaces)
        configurations = dict.fromkeys(range(self.max_rounds + 1))
        processes = []
        for bracket_num in range(self.max_rounds, -1, -1):
            n = math.ceil(
                (self.total_epochs/self.max_epochs) *
                (self.factor**bracket_num/(bracket_num+1))
            )
            configurations[bracket_num] = self.get_hyperparameter_configuration(
                cat_hparam, num_hparam, layers_hparam, combinations, n)

        torch.multiprocessing.set_start_method('spawn', force=True)
        with MyPool(self.n_device) as p:
            list_best_config = p.starmap(self.fit_multiple, [(
                X, y, configurations[bracket_num], bracket_num) for bracket_num in range(self.max_rounds, -1, -1)])

        best_config = pd.DataFrame(list_best_config)
        best_config = best_config.sort_values(
            ['score'], ascending=False).reset_index(drop=True).head(1)

        self.best_config = best_config
        self.best_params_ = best_config['hparams']
        self.best_score_ = best_config['score']
