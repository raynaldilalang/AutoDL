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

from copy import copy, deepcopy
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

class GATorchSearchCV:
    def __init__(self, estimator, search_spaces, scoring, max_epochs, population_size,                 
                 crossover_rate=0.5, mutation_rate=0.01, n_generations=10,
                 cv=3, random_state=420, greater_is_better=True,
                 n_jobs_model=1, n_jobs_cv=1, device='cpu', gpu_ids=None,
                 log_path='./'):
        self.log_path = log_path
        logging.basicConfig(
            filename=self.log_path, level=logging.INFO, filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.scoring = scoring
        self.max_epochs = max_epochs
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.cv = cv
        self.random_state = random_state
        self.greater_is_better = greater_is_better

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
            f'Initializing Torch GA Search using {self.n_device} {self.device} devices')

    @staticmethod
    def get_mean_cv_score(model, X, y, scoring, cv, n_jobs, verbose, model_id=None):
        start = time.time()
        if model_id != None:
            logging.info(f'Starting CV on model {model_id}')
        mean_cv_score = cross_val_score_torch(model, X, y, scoring, cv, n_jobs, verbose).mean()
        end = time.time()
        process_time = pd.Timedelta(end-start, unit="s")
        if model_id != None:
            logging.info(f'Finished CV on model {model_id} in {process_time}')
        return mean_cv_score

    @staticmethod
    def create_combinations(dict_hparam):
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
            configuration[ind] = dict.fromkeys(['hparams', 'score', 'isTrained'])
            configuration[ind]['score'] = -np.inf
            configuration[ind]['isTrained'] = False
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

    def _run_generation(self, X, y, gen_num, cat_hparam, num_hparam, layers_hparam, combinations):
        np.random.seed(seed=self.random_state)
        self.population[gen_num] = self.population[gen_num-1].copy()

        # Selection
        weights = np.array([self.population[gen_num - 1][n]['score']
            for n in range(self.population_size)])
        prob = weights / weights.sum()
        size_selected = int((1 - self.crossover_rate) * self.population_size)
        selected_idx = np.random.choice(
            range(self.population_size), replace=False, size=size_selected, p=prob)

        # Breeding and Mutation
        children_idx = [x for x in range(self.population_size) if x not in selected_idx]
        pairs = self.breed(selected_idx, children_idx)
        for pair in pairs.keys():
            self.population[gen_num][pair]['score'] = -np.inf
            self.population[gen_num][pair]['isTrained'] = False
            for hparam in self.population[gen_num][pair]['hparams']:
                gene_choice = random.choices(
                    pairs[pair],
                    weights=[
                        self.population[gen_num][pairs[pair][0]]['score'],
                        self.population[gen_num][pairs[pair][1]]['score']
                    ]
                )[0]
                self.population[gen_num][pair]['hparams'][hparam] = self.population[gen_num][gene_choice]['hparams'][hparam]

            for layer_num in range(len(self.population[gen_num][pair]['hparams']['list_hidden_layer'])):
                isMutate = random.choices([False, True], weights=[1-self.mutation_rate, self.mutation_rate])
                if isMutate:
                    if len(layers_hparam['num_neuron']) == 3:
                        try:
                            distribution = eval(
                                layers_hparam['num_neuron'][2].replace("-", ""))
                            self.population[gen_num][pair]['hparams']['list_hidden_layer'][layer_num] = int(distribution.rvs(
                                layers_hparam['num_neuron'][0], layers_hparam['num_neuron'][1]-layers_hparam['num_neuron'][0]))
                        except NameError:
                            logging.warning(
                                f'WARNING: Distribution {layers_hparam["num_neuron"][2]} not found, generating random number uniformly.')
                            self.population[gen_num][pair]['hparams']['list_hidden_layer'][layer_num] = randint.rvs(
                                layers_hparam['num_neuron'][0], layers_hparam['num_neuron'][1]+1)
                    else:
                        self.population[gen_num][pair]['hparams']['list_hidden_layer'][layer_num] = randint.rvs(
                            layers_hparam['num_neuron'][0], layers_hparam['num_neuron'][1]+1)

            for hparam in cat_hparam:
                isMutate = random.choices([False, True], weights=[1-self.mutation_rate, self.mutation_rate])
                if isMutate:
                    self.population[gen_num][pair]['hparams'][hparam] = random.choices(cat_hparam[hparam])[0]

            for hparam in num_hparam:
                isMutate = random.choices([False, True], weights=[1-self.mutation_rate, self.mutation_rate])
                if isMutate:
                    if len(num_hparam[hparam]) == 3:
                        try:
                            distribution = eval(
                                num_hparam[hparam][2].replace("-", ""))
                            if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                                self.population[gen_num][pair]['hparams'][hparam] = int(distribution.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0]))
                            else:
                                self.population[gen_num][pair]['hparams'][hparam] = distribution.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])
                        except NameError:
                            logging.warning(
                                f'WARNING: Distribution {num_hparam[hparam][2]} not found, generating random number uniformly.')
                            if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                                self.population[gen_num][pair]['hparams'][hparam] = randint.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]+1)
                            else:
                                self.population[gen_num][pair]['hparams'][hparam] = uniform.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])
                    else:
                        if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                            self.population[gen_num][pair]['hparams'][hparam] = randint.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]+1)
                        else:
                            self.population[gen_num][pair]['hparams'][hparam] = uniform.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])

        self._fit_generation(X, y, gen_num)

    @staticmethod
    def create_model(model_name, random_state, epoch, device, **hparams):
        model = eval(f'{model_name}')(
            **hparams, epoch=int(epoch), random_state=random_state, device=device)

        return model

    @staticmethod
    def get_top_k(leaderboard, k):
        leaderboard = leaderboard.copy()
        configurations = pd.DataFrame.from_dict(leaderboard, orient='index')
        configurations = configurations.sort_values(
            ['score'], ascending=False).reset_index(drop=True).head(k)
        configurations = configurations.to_dict(orient='index')

        return configurations

    @staticmethod
    def breed(parents, children):
        pairs = {}
        for child in children:
            pairs[child] = (random.choice(parents), random.choice(parents))

        return pairs

    def _fit_multiple(self, queue):
        list_model = queue.get()
        torch.multiprocessing.set_start_method('spawn', force=True)
        with MyPool(self.n_jobs_model) as p:
            list_score = p.starmap(self.get_mean_cv_score, list_model)
        queue.put(list_score)

    def _fit_generation(self, X, y, gen_num):
        start = time.time()
        logging.info(f'Fitting models for generation {gen_num}')

        dict_toTrain_model_by_device = {device_num: [] for device_num in range(self.n_device)}
        dict_toTrain_score_by_device = {device_num: [] for device_num in range(self.n_device)}
        dict_toTrain_idx_by_device = {device_num: [] for device_num in range(self.n_device)}
        trained_idx = 0
        for chromosome in range(self.population_size):
            device_used = self.device
            if not self.population[gen_num][chromosome]['isTrained']:
                if device_used == 'cuda':
                    device_num = self.gpu_ids[trained_idx % self.n_device]
                    device_used += f':{device_num}'
                else:
                    device_num = 0
                model = self.create_model(
                    self.estimator,
                    random_state=self.random_state,
                    epoch=self.max_epochs,
                    device=device_used,
                    **self.population[gen_num][chromosome]['hparams']
                )
                verbose = 0
                dict_toTrain_model_by_device[device_num].append(
                    (model, X, y, self.scoring, self.cv, self.n_jobs_cv, verbose, chromosome))
                dict_toTrain_idx_by_device[device_num].append(chromosome)
                trained_idx += 1

        num_trained = trained_idx
        processes = []
        queues = []
        for device_num in range(self.n_device):
            q = Queue()
            q.put(dict_toTrain_model_by_device[device_num])
            p = Process(
                target=self._fit_multiple,
                args=(q, )
            )
            processes.append(p)
            queues.append(q)
            p.start()

        for process in processes:
            process.join()

        for device_num in range(self.n_device):
            dict_toTrain_score_by_device[device_num] = queues[device_num].get()

        list_toTrain_model_by_device = [
            dict_toTrain_model_by_device[device_num] for device_num in range(self.n_device)
        ]
        list_toTrain_score_by_device = [
            dict_toTrain_score_by_device[device_num] for device_num in range(self.n_device)
        ]

        for trained_idx in range(num_trained):
            device_used = self.device
            if device_used == 'cuda':
                device_num = self.gpu_ids[trained_idx % self.n_device]
                device_used += f':{device_num}'
            else:
                device_num = 0
            chromosome = list_toTrain_model_by_device[device_num].pop(0)[-1]
            self.population[gen_num][chromosome]['score'] = list_toTrain_score_by_device[device_num].pop(0)
            self.population[gen_num][chromosome]['isTrained'] = True
        
        leaderboard = deepcopy(self.population[gen_num])
        best_config_by_gen = self.get_top_k(leaderboard, 1)[0]
        best_config_by_gen['hparams'].update({'epoch': self.max_epochs})
        self.list_best_config.append({'gen': gen_num, **best_config_by_gen})

        end = time.time()
        process_time = pd.Timedelta(end-start, unit="s")
        logging.info(f'Finished fitting models for generation {gen_num} in {process_time}')

    def fit(self, X, y):
        start = time.time()
        logging.info(f'Genetic Algorithm on {self.estimator} \n')
        cat_hparam, num_hparam, layers_hparam, combinations = self.create_combinations(
            self.search_spaces
        )
        self.population = {0: None}
        self.population[0] = self.get_hyperparameter_configuration(
            cat_hparam, num_hparam, layers_hparam, combinations, self.population_size
        )
        self.list_best_config = []
        self._fit_generation(X, y, 0)
        for gen_num in range(1, self.n_generations + 1):
            self._run_generation(X, y, gen_num, cat_hparam, num_hparam, layers_hparam, combinations)

        best_config = pd.DataFrame(self.list_best_config).drop(columns=['isTrained'])
        best_config = best_config.sort_values(
            ['score'], ascending=not self.greater_is_better).reset_index(drop=True)

        self.best_config = best_config

        end = time.time()
        process_time = pd.Timedelta(end-start, unit="s")
        logging.info(f'Finished Genetic Algorithm on {self.estimator} in {process_time}')

    @property
    def best_params_(self):
        return self.best_config.loc[0, 'hparams']
    
    @property
    def best_score_(self):
        return self.best_config.loc[0, 'score']
