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
from itertools import chain

import multiprocessing
import torch.multiprocessing
from torch.multiprocessing import Pool, Process, Queue

from models import *

import os
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

class HyperBandTorchSearchCV:
    """
    Hyperband optimization over deep learning hyperparameters. Used specifically for PyTorch
    neural network modules.

    It is a principled early-stoppping method that adaptively allocates a pre-defined resource,
    e.g., iterations, data samples or number of features, to randomly sampled configurations.

    Parameters
    ----------
    estimator : str.
        An object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    search_spaces : dict.
        Dictionary with hyperparameter names as keys
        1. list of hyperparameter values to randomly choose form (uniformly)
        2. tuple of size two, as the lower bound and upper bound for the hyperparameters
        to choose from (uniformly)
        3. tuple of size three, with the third argument as the distribution name

    scoring : callable.
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int.
        The number of folds in cross-validation splitting strategy.

    random_state : int.
        Seed number for random number generators

    greater_is_better : bool.
        True if greater scoring metric means better result, and False otherwise

    n_jobs : int.
        The number of parallel model fittings done in a single round of a bracket

    device : str.
        One of 'cpu' or 'cuda'

    gpu_ids : list.
        List of gpu number used (if cuda device is used)
    """
    def __init__(self, estimator, search_spaces,
                 scoring, max_epochs, factor=3,
                 cv=3, random_state=420, greater_is_better=True,
                 n_jobs_model=1, n_jobs_cv=1, device='cpu', gpu_ids=None,
                 log_path='./log.log'):
        self.log_path = log_path
        logging.basicConfig(
            filename=self.log_path, level=logging.INFO, filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.scoring = scoring
        self.max_epochs = max_epochs
        self.factor = factor
        self.cv = cv
        self.random_state = random_state
        self.greater_is_better = greater_is_better
        self.max_rounds = math.floor(math.log(max_epochs, factor))
        self.total_epochs = (self.max_rounds + 1) * max_epochs

        self.n_jobs_model = n_jobs_model
        self.n_jobs_cv = n_jobs_cv
        self.device = device

        available_cudas = torch.cuda.device_count()
        if available_cudas == 0 and device == 'cuda':
            self.device = 'cpu'
            n_gpu = 0
            logging.warning(f'WARNING: No cuda devices are found, using cpu instead')
        elif available_cudas < len(gpu_ids):
            n_gpu = available_cudas
            logging.warning(f'WARNING: Only {available_cudas} cuda devices are found')
        else:
            n_gpu = len(gpu_ids)

        self.gpu_ids = gpu_ids
        if n_gpu > 0:
            self.n_device = n_gpu
        else:
            self.n_device = 1

        logging.info(
            f'Initializing Torch HyperBand Search using {self.n_device} {self.device} devices')

    @staticmethod
    def get_mean_cv_score(model, X, y, scoring, cv, n_jobs, verbose):
        """
        Returns the mean cross-validation score of some pytorch model

        Parameters
        ----------
        model : callable.
            Callable object with method `fit`

        X : array-like.
            Array of predictors

        y : array-like.
            Array of target/labels

        scoring : callable.
            A scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        n_jobs : int.
            Number of parallel processes used in cross-validation

        verbose : int
            Has value 1 if epoch progress bar should be shown

        Returns
        -------
        mean_score : float.
            Mean cross-validation score

        """
        
        mean_cv_score = cross_val_score_torch(model, X, y, scoring, cv, n_jobs, verbose).mean()
        return mean_cv_score

    @staticmethod
    def create_combinations(dict_hparam):
        """
        Separate the search space into categorical, numerical, and NN-layer hyperparameters;
        as well as generate all posible combinations of categorical hyperparameters

        Parameters
        ----------
        dict_hparam : dict.
            search space given in __init__

        Returns
        -------
        cat_hparam : dict.
            Dictionary of categorical hyperparameters

        num_hparam : dict.
            Dictionary of numerical hyperparameters

        layers_hparam : dict.
            Dictionary of NN-layers hyperparameters

        combinations : pandas.DataFrame.
            List of all possible combinations of categorical hyperparameters

        """
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
        """
        Generates n hyperparameter configurations based on the given dictionaies of hyperparameters

        Parameters
        -------
        cat_hparam : dict.
            Dictionary of categorical hyperparameters

        num_hparam : dict.
            Dictionary of numerical hyperparameters

        layers_hparam : dict.
            Dictionary of NN-layers hyperparameters

        combinations : pandas.DataFrame.
            List of all possible combinations of categorical hyperparameters

        Returns
        -------
        configuration : dict,
            Dictionary with the configuration number as the outmost key,
            and the hyperparameters as the inner key
        """
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
                    logging.warning(
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
                    logging.warning(
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
                            logging.warning(
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

        return configuration

    def _create_brackets(self):
        """
        Generate hyperband brackets, which contains the number of models produced
        in each round on each bracket

        Updates
        -------
        brackets : dict
            Dictionary of brackets. Each bracket is a dictionary containing the number
            of models to train ('ni') and the resources allocated on each one ('ri') 
        """
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
            logging.info(f'Bracket {bracket_num} setup:')
            logging.info(bracket_df.rename(
                columns={'ni': 'Number of Configurations', 'ri': 'Resource'}))

        self.brackets = brackets

    @staticmethod
    def create_model(model_name, random_state, epoch, device, log_path, **hparams):
        """
        Initiate a model object

        Parameters
        ----------
        model_name : str
            An object of that type is instantiated.
            This object is assumed to implement the scikit-learn estimator api.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        random_state : int.
            Seed number for random number generators

        epoch : int.
            The number times that the learning algorithm will work
            through the entire training dataset

        device : str.
            One of 'cpu' or 'cuda'

        Returns
        -------
        model : callable.
            Callable object with method `fit`
        """
        model = eval(f'{model_name}')(
            **hparams, epoch=int(epoch), random_state=random_state, device=device,
            log_path=log_path
        )

        return model

    @staticmethod
    def get_top_k(leaderboard, k):
        """
        Gets the top k configurations in one round of a bracket

        Parameters
        ----------
        leaderboard : dict
            Dictionary of all configurations in one round of a bracket,
            together with their scores

        k : int.
            Number of configurations to get

        Returns
        -------
        model : callable.
            Callable object with method `fit`
        """
        configurations = pd.DataFrame.from_dict(leaderboard, orient='index')
        configurations = configurations.sort_values(
            ['score'], ascending=False).reset_index(drop=True).head(k)
        configurations = configurations.to_dict(orient='index')

        return configurations

    def _fit_multiple(self, X, y, configurations, bracket_num):
        """
        Fits all configurations in one round of a bracket, and
        returns the best configuration of that round

        Parameters
        ----------
        bracket_round : dict
            Dictionary of all configurations in one round of a bracket,
            together with their scores

        k : int.
            Number of configurations to get

        Returns
        -------
        best_config_by_round : callable.
            Callable object with method `fit`
        """
        device_used = self.device
        if device_used == 'cuda':
            device_used += f':{self.gpu_ids[bracket_num % self.n_device]}'
        list_toTrain_model = []
        best_config_by_round = []

        for i in tqdm(range(bracket_num + 1), desc=f'Bracket {bracket_num}', position=(self.max_rounds-bracket_num), leave=True):
            for contender in range(self.brackets[bracket_num][i]['ni']):
                self.brackets[bracket_num][i]['contenders'][contender] = dict.fromkeys([
                                                                                       'hparams', 'score'])
                self.brackets[bracket_num][i]['contenders'][contender]['hparams'] = configurations[contender]['hparams']
                model = self.create_model(
                    self.estimator,
                    random_state=self.random_state,
                    epoch=self.brackets[bracket_num][i]['ri'],
                    device=device_used,
                    log_path=self.log_path,
                    **configurations[contender]['hparams']
                )
                verbose = 0
                list_toTrain_model.append(
                    (model, X, y, self.scoring, self.cv, self.n_jobs_cv, verbose))

            torch.multiprocessing.set_start_method('spawn', force=True)
            with MyPool(self.n_jobs_model) as p:
                list_toTrain_score = p.starmap(
                    self.get_mean_cv_score, list_toTrain_model)

            for contender in range(self.brackets[bracket_num][i]['ni']):
                self.brackets[bracket_num][i]['contenders'][contender]['score'] = list_toTrain_score[contender]

            configurations = self.get_top_k(
                self.brackets[bracket_num][i]['contenders'],
                k=max(math.floor(
                    self.brackets[bracket_num][i]['ni']/self.factor), 1)
            )

            best_config = configurations[0].copy()
            best_config_by_round.append({
                'bracket': bracket_num,
                'round': i,
                'epoch': int(self.brackets[bracket_num][i]['ri']),
                **best_config
            })

        return best_config_by_round

    def _fit_in_pool(self, queue):
        X, y, configurations, list_bracket_num = queue.get()
        torch.multiprocessing.set_start_method('spawn', force=True)
        with MyPool(1) as p:
            list_best_config = p.starmap(self._fit_multiple, [(
                X, y, configurations[bracket_num], bracket_num) for bracket_num in list_bracket_num
        ])
        queue.put(list_best_config)
        

    def fit(self, X, y):
        """
        Executes hyperband search based on the given configurations

        Parameters
        ----------
        X : array-like.
            Array of predictors

        y : array-like.
            Array of target/labels

        Updates
        -------
        best_config : pandas.DataFrame.
            Dataframe which list all the best configurations from each round of each bracket
        """
        start = time.time()
        logging.info(f'HyperBand on {self.estimator} \n')
        self._create_brackets()
        cat_hparam, num_hparam, layers_hparam, combinations = self.create_combinations(
            self.search_spaces)
        configurations = dict.fromkeys(range(self.max_rounds + 1))
        dict_bracket_by_device = {device_num: [] for device_num in range(self.n_device)}
        dict_best_config_by_device = {device_num: [] for device_num in range(self.n_device)}

        for bracket_num in range(self.max_rounds, -1, -1):
            device_used = self.device
            if device_used == 'cuda':
                device_num = self.gpu_ids[bracket_num % self.n_device]
                device_used += f':{device_num}'
            else:
                device_num = 0
            n = math.ceil(
                (self.total_epochs/self.max_epochs) *
                (self.factor**bracket_num/(bracket_num+1))
            )
            configurations[bracket_num] = self.get_hyperparameter_configuration(
                cat_hparam, num_hparam, layers_hparam, combinations, n)
            dict_bracket_by_device[device_num].append(bracket_num)
            
        processes = []
        queues = []
        for device_num in range(self.n_device):
            q = Queue()
            q.put((X, y, configurations, dict_bracket_by_device[device_num]))
            p = Process(
                target=self._fit_in_pool,
                args=(q, )
            )
            processes.append(p)
            queues.append(q)
            p.start()

        for process in processes:
            process.join()

        for device_num in range(self.n_device):
            dict_best_config_by_device[device_num] = queues[device_num].get()

        logging.info(dict_bracket_by_device)
        logging.info(dict_best_config_by_device)

        list_best_config = [
            dict_best_config_by_device[device_num] for device_num in range(self.n_device)
        ]
        list_best_config = unnest(list_best_config, repeat=2)

        best_config = pd.DataFrame(list_best_config)
        best_config = best_config.sort_values(
            ['score'], ascending=not self.greater_is_better).reset_index(drop=True)

        self.best_config = best_config
        end = time.time()
        process_time = pd.Timedelta(end-start, unit="s")
        logging.info(f'\nFinished Genetic Algorithm on {self.estimator} in {process_time}')

    @property
    def best_params_(self):
        return self.best_config.loc[0, 'hparams']
    
    @property
    def best_score_(self):
        return self.best_config.loc[0, 'score']
