"""This module contains the standard active learner used in ELFI."""

import numpy as np

from elfi.methods.bo.acquisition import LCBSC, AcquisitionBase
from elfi.methods.bo.active_learner import ActiveLearner
from elfi.methods.bo.gpy_regression import GPyRegression


class BoWrapper(ActiveLearner):
    """Active learner that uses the ELFI acquisition functions."""

    def __init__(self,
                 parameter_names,
                 bounds=None,
                 target_model=None,
                 acquisition_method=None,
                 acq_noise_var=0,
                 exploration_rate=10,
                 seed=None,
                 ):
        """Initialize active learner.

        Parameters
        ----------
        parameter_names : List[str]
            Input parameter names.
        bounds : Dict[str, Tuple[float, float]], optional
            Lower and upper bound for each parameter.
        target_model : GPyRegression, optional
            Gaussian process model.
        acquisition_method : AcquisitionBase, optional
            Method used to calculate acquisition scores. Defaults to LCBSC.
        acq_noise_var : float or Dict[str, float], optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
        exploration_rate : float, optional
            Exploration rate used in the acquisition method.
        seed : int, optional

        """
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
        """Update model with new evidence.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize model fit.

        """
        if self.target_model.n_evidence > 0:
            self.target_model.update(x, y, optimize)
        else:
            self.init_x.append(x)
            self.init_y.append(y)
            if optimize:
                self._init_model()

    def acquire(self, n, t=None):
        """Return the next batch of acquisition points.

        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acquisition batch index (starting from 0).

        Returns
        -------
        np.array
            with shape (n, input_dim)

        """
        return self.acquisition_method.acquire(n, t=t)

    def get_model(self):
        """Return current model fit.

        Returns
        -------
        GPyRegression or None

        """
        if self.target_model.n_evidence > 0:
            return self.target_model
        else:
            return None

    def evaluate_acquisition_function(self, x, t=None):
        """Return the acquisition function value at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
        t : int
            current acquisition batch index (starting from 0)

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        if self.target_model.n_evidence > 0:
            return self.acquisition_method.evaluate(x, t=t)
        else:
            return np.zeros((x.shape[0], 1))

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
