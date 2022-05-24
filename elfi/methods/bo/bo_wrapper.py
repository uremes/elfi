"This module contains interface for active learning."

import logging

import matplotlib.pyplot as plt
import numpy as np

from elfi.methods.bo.acquisition import AcquisitionBase, LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression

logger = logging.getLogger(__name__)

class ActiveLearnerBase():

    # essential:

    def update(self, x, y, optimize=True):
        pass

    def acquire(self, n, t=None):
        pass

    def get_model(self):
        pass

    # extras/convenience:

    def get_acquisition_value(self, x, t=None):
        pass

    def get_model_pred(self, x):
        pass

    def get_evidence(self):
        pass

class BoWrapper():
    """
    Active learner that uses the ELFI acquisition functions.
    """

    def __init__(self,
                 parameter_names,
                 bounds=None,
                 target_model=None,
                 acquisition_method=None,
                 acq_noise_var=0,
                 exploration_rate=10,
                 seed=None,
                 ):

        self.parameter_names = parameter_names
        self.seed = seed
        self.target_model = self._resolve_target_model(target_model, bounds)
        self.bounds = self.target_model.bounds
        self.input_dim = len(bounds)
        self.acquisition_method = self._resolve_acquisition_method(acquisition_method,
                                                                   acq_noise_var,
                                                                   exploration_rate)

        self.init_x = []
        self.init_y = []

    def update(self, x, y, optimize=True):

        if self.target_model.n_evidence > 0:
            self.target_model.update(x, y, optimize)
        else:
            self.init_x.append(x)
            self.init_y.append(y)
            if optimize:
                self._init_model();

    def acquire(self, n, t=None):
        
        return self.acquisition_method.acquire(n, t=t)

    def get_model(self):

        if self.target_model.n_evidence > 0:
            return self.target_model
        else:
            return None

    def get_acquisition_value(self, x, t=None):

        return self.acquisition_method.evaluate(x, t=t)

    def get_model_pred(self, x, observation_noise=True):

        return self.target_model.predict(x, noiseless=not(observation_noise))

    def get_evidence(self):

        return self.target_model.X, self.target_model.Y

    def _resolve_target_model(self, target_model, bounds):

        if target_model is None:
            return GPyRegression(self.parameter_names, bounds=bounds)
        if isinstance(target_model, GPyRegression):
            return target_model
        raise TypeError('target_model must be an instance of GPyRegression.')

    def _resolve_acquisition_method(self, acquisition_method, acq_noise_var, exploration_rate):

        if acquisition_method is None:
            return LCBSC(self.target_model,
                         noise_var=acq_noise_var,
                         exploration_rate=exploration_rate,
                         seed=self.seed)
        if isinstance(acquisition_method, AcquisitionBase):
            return acquisition_method
        raise TypeError('acquisition_method must be an instance of AcquisitionBase.')

    def _init_model(self):

        x = np.array(self.init_x).reshape(-1, self.input_dim)
        y = np.array(self.init_y).reshape(-1, 1)
        self.target_model.update(x, y, optimize=True)
