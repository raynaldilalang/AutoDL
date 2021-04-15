import numpy as np
import pandas as pd
import random
import math
from itertools import product
from scipy.stats import *


class HyperBandSearchCV:
    def __init__(self, estimator, search_spaces,
                 scoring, max_epochs, factor=3,
                 cv=1, random_state=420):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.scoring = scoring
        self.max_epochs = max_epochs
        self.factor = factor
        self.random_state = random_state
        self.max_rounds = math.log(max_epochs, factor)
        self.total_epochs = (self.max_rounds + 1) * max_epochs

    @staticmethod
    def create_cat_combinations(dict_hparam):
        cat_hparam = {}
        num_hparam = {}

        for hparam in dict_hparam.keys():
            if type(dict_hparam[hparam]) == list:
                cat_hparam[hparam] = dict_hparam[hparam]
            else:
                num_hparam[hparam] = dict_hparam[hparam]

        combinations_product = product(
            *(cat_hparam[hparam] for hparam in cat_hparam.keys()))
        combinations = pd.DataFrame(
            combinations_product, columns=cat_hparam.keys())

        return cat_hparam, num_hparam, combinations

    @staticmethod
    def get_hyperparameter_configuration(cat_hparam, num_hparam, combinations, n, random_state=420):
        np.random.seed(seed=random_state)
        population = dict.fromkeys(range(n))
        for ind in range(n):
            population[ind] = {'hparams': None}
            population[ind]['hparams'] = dict.fromkeys(
                [*cat_hparam.keys(), *num_hparam.keys()])

            if len(cat_hparam):
                cat_combination_num = random.randint(
                    0, len(combinations)-1)
                for hparam in cat_hparam.keys():
                    population[ind]['hparams'][hparam] = combinations.loc[cat_combination_num, hparam]

            if len(num_hparam):
                for hparam in num_hparam.keys():
                    if len(num_hparam[hparam]) == 3:
                        try:
                            distribution = eval(
                                num_hparam[hparam][2].replace("-", ""))
                            population[ind]['hparams'][hparam] = distribution.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])
                        except NameError:
                            print(
                                f'WARNING: Distribution {num_hparam[hparam][2]} not found, generating random number uniformly.')
                            if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                                population[ind]['hparams'][hparam] = randint.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]+1)
                            else:
                                population[ind]['hparams'][hparam] = uniform.rvs(
                                    num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])
                    else:
                        if (type(num_hparam[hparam][0]) == int) and (type(num_hparam[hparam][1]) == int):
                            population[ind]['hparams'][hparam] = randint.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]+1)
                        else:
                            population[ind]['hparams'][hparam] = uniform.rvs(
                                num_hparam[hparam][0], num_hparam[hparam][1]-num_hparam[hparam][0])

            population[ind]['isTrained'] = False

        return population

    def create_brackets(self):
        brackets = dict.fromkeys(range(self.max_rounds + 1))
        for bracket_num in brackets:
            n = ceil(
                (self.total_epochs/self.max_epochs) *
                (self.factor**bracket_num/(bracket_num+1))
            )
            r = self.max_epochs * self.factor**(-bracket_num)
            brackets[bracket_num] = dict.fromkeys(range(bracket_num))
            for i in range(bracket_num):
                brackets[bracket_num][i]['ni'] = floor(n * self.factor**(-i))
                brackets[bracket_num][i]['ri'] = r * self.factor**i
                brackets[bracket_num][i]['contenders'] = dict.fromkeys(
                    range(brackets[bracket_num][i]['ni']))

        self.brackets = brackets

    @staticmethod
    def create_model(model_name, random_state, epoch, **hparams):
        try:
            model = eval(f'{model_name}')(
                **hparams, epoch=epoch, random_state=random_state)
        except TypeError:
            model = eval(f'{model_name}')(**hparams, epoch=epoch)

        return model
    
    @staticmethod
    def top_k(bracket_round, k):
        configurations = pd.from_dict(bracket_round, orient='index')
        configurations = configurations.sort_values(
            ['score'], ascending=False).reset_index(drop=True).head(k)
        configurations = pd.to_dict(configurations, orient='index')
        
        return configurations
        
    def fit(self, X, y):
        cat_hparam, num_hparam, combinations = self.create_cat_combinations(
            self.search_spaces)
        for bracket_num in self.brackets:
            n = ceil(
                (self.total_epochs/self.max_epochs) *
                (self.factor**bracket_num/(bracket_num+1))
            )
            configurations = self.get_hyperparameter_configuration(
                cat_hparam, num_hparam, combinations, n)
            for i in range(bracket_num):
                for contender in range(brackets[bracket_num][i]['ni']):
                    self.brackets[bracket_num][i]['contenders'][contender] = dict.fromkeys(['hparams', 'score'])
                    self.brackets[bracket_num][i]['contenders'][contender]['hparam'] = configurations[i]
                    model = create_model(
                        'MLP',
                        random_state=420,
                        epoch=self.brackets[bracket_num][i]['ri'],
                        **configurations[i]
                    )
                    self.brackets[bracket_num][i]['contenders'][contender]['score'] = cross_val_score_torch(
                        model, X, y,
                        scoring=self.scoring,
                        device=self.device,
                        cv=self.cv
                    )
                configurations = top_k(
                    self.brackets[bracket_num][i]['contenders'],
                    k=floor(brackets[bracket_num][i]['ni']/self.factor)
                )
