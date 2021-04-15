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

import multiprocessing
from multiprocessing import Pool

import Models

class HyperBandTorchSearchCV:
    def __init__(self, estimator, search_spaces,
                 scoring, max_epochs, factor=3,
                 cv=1, random_state=420,
                 n_jobs=1):
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

    def get_score(self, value):
        return cross_val_score_torch(**value).mean()

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
                [*cat_hparam.keys(), *num_hparam.keys()])

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
        for bracket_num in brackets:
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
                brackets[bracket_num][i]['ri'] = r * self.factor**i
                brackets[bracket_num][i]['contenders'] = dict.fromkeys(
                    range(brackets[bracket_num][i]['ni']))

        self.brackets = brackets

    @staticmethod
    def create_model(model_name, random_state, epoch, **hparams):
        try:
            model = eval(f'{model_name}')(**hparams, epoch=epoch)
        except:
            try:
                model = eval(f'{model_name}')(**hparams, max_iter=epoch)
            except:
                model = eval(f'{model_name}')(**hparams, max_depth=epoch)

        return model

    @staticmethod
    def get_top_k(bracket_round, k):
        configurations = pd.DataFrame.from_dict(bracket_round, orient='index')
        configurations = configurations.sort_values(
            ['score'], ascending=False).reset_index(drop=True).head(k)
        configurations = configurations.to_dict(orient='index')

        return configurations

    def fit_multiple(self, X, y, configurations, bracket_num, i):
        list_toTrain_model = []
        for contender in range(self.brackets[bracket_num][i]['ni']):
            self.brackets[bracket_num][i]['contenders'][contender] = dict.fromkeys(['hparams', 'score'])
            self.brackets[bracket_num][i]['contenders'][contender]['hparams'] = configurations[bracket_num][contender]['hparams']
            model = self.create_model(
                self.estimator,
                random_state=self.random_state,
                epoch=self.brackets[bracket_num][i]['ri'],
                **configurations[bracket_num][contender]['hparams']
            )
            list_toTrain_model.append({
                'estimator': model,
                'X': X, 'y': y,
                'scoring': self.scoring,
                'cv': self.cv,
                'n_jobs': self.cv
            })

        p = Pool(self.n_jobs)
        print(f'Starting multiprocess ({self.n_jobs})')
        with p:
            list_toTrain_score = p.map(self.get_score, list_toTrain_model)

        for contender in range(self.brackets[bracket_num][i]['ni']):
            self.brackets[bracket_num][i]['contenders'][contender]['score'] = list_toTrain_score[contender]

    def fit(self, X, y):
        print(f'HyperBand on {self.estimator}')
        self.create_brackets()
        cat_hparam, num_hparam, layers_hparam, combinations = self.create_cat_combinations(
            self.search_spaces)
        configurations = dict.fromkeys(range(self.max_rounds + 1))
        for bracket_num in range(self.max_rounds, -1, -1):
            n = math.ceil(
                (self.total_epochs/self.max_epochs) *
                (self.factor**bracket_num/(bracket_num+1))
            )
            configurations[bracket_num] = self.get_hyperparameter_configuration(
                cat_hparam, num_hparam, layers_hparam, combinations, n)
            for i in range(bracket_num + 1):
                print('Bracket', str(bracket_num)+'.'+str(i))                
                self.fit_multiple(X, y, configurations, bracket_num, i)

                configurations[bracket_num] = self.get_top_k(
                    self.brackets[bracket_num][i]['contenders'],
                    k=max(math.floor(
                        self.brackets[bracket_num][i]['ni']/self.factor), 1)
                )
            configurations[bracket_num] = configurations[bracket_num][0]
            print(f'Best of bracket {bracket_num}:',
                  configurations[bracket_num])

        best_config = pd.DataFrame.from_dict(configurations, orient='index')
        best_config = best_config.sort_values(
            ['score'], ascending=False).reset_index(drop=True).loc[0, :]

        self.best_params_ = best_config['hparams']
        self.best_score_ = best_config['score']
