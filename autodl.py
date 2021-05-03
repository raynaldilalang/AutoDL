from evolution import GATorchSearchCV
from hyperband import HyperBandTorchSearchCV
from models import MLPClassifier, MLPRegressor

class AutoDL:
    def __init__(self, task, search_spaces, max_epochs,
                 methods=['HyperBandTorchSearchCV'],
                 method_dict=None, scoring=None,
                 device='cpu', gpu_ids=None,
                 greater_is_better=True,
                 log_path='./log.log'):
        self.task = task
        self.search_spaces = search_spaces
        self.methods = methods
        self.scoring = scoring
        self.max_epochs = max_epochs
        self.greater_is_better = greater_is_better
        self.device = device
        self.gpu_ids = gpu_ids
        self.log_path = log_path

        if self.task == 'classification':
            self.estimator = 'MLPClassifier'
        elif self.task == 'regression':
            self.estimator = 'MLPRegressor'

        if method_dict == None:
            self.method_dict = {
                'HyperBandTorchSearchCV': {
                    'factor': 2,
                    'cv': 5,
                    'random_state': 420,
                    'n_jobs_model': 1,
                    'n_jobs_cv': 1,
                },
                'GATorchSearchCV': {
                    'population_size': self.max_epochs,
                    'crossover_rate': 0.2,
                    'mutation_rate': 0.001,
                    'n_generations': 10,
                    'cv': 5,
                    'random_state': 420,
                    'n_jobs_model': 1,
                    'n_jobs_cv': 1,
                }
            }
        else:
            self.method_dict = method_dict


    def fit(self, X, y):
        self.dict_searchcv = dict.fromkeys(self.methods)
        for search in self.dict_searchcv:
            search_class = eval(search)
            self.dict_searchcv[search] = search_class(
                estimator=self.estimator,
                search_spaces=self.search_spaces,
                scoring=self.scoring,
                max_epochs=self.max_epochs,
                device=self.device,
                gpu_ids=self.gpu_ids,
                greater_is_better=self.greater_is_better,
                log_path=self.log_path,
                **self.method_dict[search]
            )
            self.dict_searchcv[search].fit(X, y)

        self.best_score_ = self.dict_searchcv[self.methods[0]].best_score_
        self.best_params_ = self.dict_searchcv[self.methods[0]].best_params_
        self.best_method_ = self.methods[0]
        for search in self.dict_searchcv:
            if self.best_score_ < self.dict_searchcv[search].best_score_:
                self.best_score_ = self.dict_searchcv[search].best_score_
                self.best_params_ = self.dict_searchcv[search].best_params_
                self.best_method_ = search

        estimator_class = eval(self.estimator)
        self.best_model_ = estimator_class(**self.best_params_)
        self.best_model_.fit(X, y)     

    def predict(self, X):
        prediction = self.best_model_.predict(X)

        return prediction
        
