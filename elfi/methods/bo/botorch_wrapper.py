"This module contains wrappers for using BoTorch in ELFI."

import copy
import contextlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import AnalyticAcquisitionFunction, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

class BoTorchWrapper():

    def __init__(self,
                 parameter_names,
                 bounds,
                 model_class=None,
                 model_options=None,
                 mll_class=None,
                 mll_options=None,
                 acq_class=None,
                 acq_options=None,
                 optim_params=None,
                 scale = 1,
                 seed = None
                 ):

        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = self._resolve_bounds(bounds, self.parameter_names)
        self.model_class = model_class or SingleTaskGP
        self.model_options = model_options or self._get_default_options(self.bounds)
        self.mll_class = mll_class or ExactMarginalLogLikelihood
        self.mll_options = mll_options or {}
        self.acq_class = acq_class or LCBSC
        self.acq_options = acq_options or {'t': None, 'exploration_rate': 10}
        self.optim_params = optim_params or {'num_restarts': 5, 'raw_samples': 20}
        self.scale = scale
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # TODO add device

        self.init_x = []
        self.init_y = []

        self.model = None

    def update(self, x, y, optimize=True):

        y = self.scale * y

        if self.model is not None:
            x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
            y = torch.tensor(y, dtype=torch.double).reshape(-1, 1)
            self.train_x = torch.cat([self.train_x, x])
            self.train_y = torch.cat([self.train_y, y])
            self._init_model(self.train_x, self.train_y, state_dict=self.model.state_dict())
        else:
            self.init_x.append(x)
            self.init_y.append(y)
            if optimize:
                self.train_x = np.array(self.init_x).reshape(-1, self.input_dim)
                self.train_y = np.array(self.init_y).reshape(-1, 1)
                self.train_x = torch.tensor(self.train_x, dtype=torch.double)
                self.train_y = torch.tensor(self.train_y, dtype=torch.double)
                self._init_model(self.train_x, self.train_y)

        if optimize:
            self._optimize()

    def acquire(self, n, t=None):

        if n > 1 and issubclass(self.acq_class, AnalyticAcquisitionFunction):
            raise ValueError('Selected acquisition class does not work with batch size > 1.')

        # update acquisition index
        if 't' in self.acq_options: self.acq_options['t'] = t

        acq_function = self.acq_class(self.model, **self.acq_options)
        x, value = optimize_acqf(acq_function, bounds=self.bounds, q=n, **self.optim_params)

        return x.numpy()

    def get_model(self):

        if self.model is not None:
            model = copy.deepcopy(self.model)
            scale = 1 / self.scale
            bounds = list(np.transpose(self.bounds.numpy()))
            return GPyTorchRegression(model, self.parameter_names, bounds, scale=scale)
        else:
            return None

    def get_acquisition_value(self, x, t=None):

        x = torch.tensor(x, dtype=torch.double).reshape(-1, 1, self.input_dim)

        # update acquisition index
        if 't' in self.acq_options: self.acq_options['t'] = t

        acq_function = self.acq_class(self.model, **self.acq_options)
        return acq_function(x).detach().numpy()

    def get_model_pred(self, x, observation_noise=True, fast_pred_var=True):

        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)

        # activate evaluation mode
        self.model.eval()
        self.model.likelihood.eval()

        # resolve computation context
        if fast_pred_var:
            computation_context = gpytorch.settings.fast_pred_var()
        else:
            computation_context = contextlib.nullcontext()

        with torch.no_grad(), computation_context:
            pred = self.model.posterior(x, observation_noise=observation_noise)

        return pred.mean.numpy().reshape(-1, 1), pred.variance.detach().numpy().reshape(-1, 1)

    def get_evidence(self):

        return self.train_x.numpy(), self.train_y.numpy()

    def _resolve_bounds(self, bounds, parameter_names):

        if isinstance(bounds, dict):
            # check that all parameters have bounds
            for param in parameter_names:
                if param not in bounds:
                    raise ValueError(f'Parameter \'{param}\' not found in bounds.')
            # convert bounds to botorch format
            bounds = [bounds[param] for param in parameter_names]
            return torch.tensor(np.transpose(bounds), dtype=torch.double)
        else:
            return bounds

    def _get_default_options(self, bounds):

        options = {}

        # use the active learner bounds to normalise inputs
        if not(all(bounds[0] == 0) and all(bounds[1] == 1)):
            options['input_transform'] =  Normalize(bounds.shape[1], bounds=bounds)

        # standardise outcome mean and variance
        options['outcome_transform'] = Standardize(1)

        return options

    def _init_model(self, x, y, state_dict=None):

        self.model = self.model_class(x, y, **self.model_options)
        if state_dict is not None: self.model.load_state_dict(state_dict)

    def _optimize(self):

        mll = self.mll_class(self.model.likelihood, self.model, **self.mll_options)
        fit_gpytorch_model(mll)


class LCBSC(UpperConfidenceBound):
    """
    Lower confidence bound selection criterion as implemented in ELFI.

    The acquisition score is calculated as `LCBSC(x) = mu(x) - sqrt(beta) * s(x)`,
    where `mu` and `s` are the posterior mean and standard deviation, and beta is
    calculated based on an exploration rate parameter and the current acquisition
    index.

    Does not support batch acquisitions.

    """
    def __init__(self, model, exploration_rate, t):
        """
        Initalize acquisition function.

        Parameters
        ----------
        model : botorch.models.model.Model
            A fitted single-outcome GP model.
        exploration_rate : float
            Exploration rate used to calculate beta. Must be positive, exploration_rate > 0.
        t : int
            Acquisition index.
        maximize: bool, optional

        """
        beta = self._beta(1/exploration_rate, model.train_inputs[0].shape[1], t)
        super().__init__(model, beta=beta, maximize=False)
        
    def _beta(self, delta, d, t):
        """Calculate beta based on the update rule used in ELFI."""
        t += 1
        return 2 * np.log(t**(2 * d + 2) * np.pi**2 / (3 * delta))


class GPyTorchRegression():

    def __init__(self, model, parameter_names, bounds, scale=1, fast_pred_var=True):

        self.model = model
        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = self._resolve_bounds(bounds, self.parameter_names)
        self.scale = scale
        if fast_pred_var:
            self.computation_context = gpytorch.settings.fast_pred_var()
        else:
            self.computation_context = contextlib.nullcontext()

        self.model.eval()
        self.model.likelihood.eval()

    def predict(self, x, noiseless=False):
        
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)

        with torch.no_grad(), self.computation_context:
            pred = self.model.posterior(x, observation_noise=not(noiseless))

        m = self.scale * pred.mean.detach().numpy().reshape(-1, 1)
        v = pred.variance.detach().numpy().reshape(-1, 1)
        return m, v
    
    def predict_mean(self, x):
        
        return self.predict(x, noiseless=True)[0]
    
    def predictive_gradients(self, x):
        
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True
        
        # define the mean and variance function that we want to differentiate
        m = lambda x: self.model.posterior(x).mean.sum()
        v = lambda x: self.model.posterior(x).variance.sum()
        
        with self.computation_context:
            dmdx  = torch.autograd.functional.jacobian(m, x)
            dvdx  = torch.autograd.functional.jacobian(v, x)

        dmdx = self.scale * dmdx.numpy().reshape(-1, self.input_dim)
        dvdx = dvdx.numpy().reshape(-1, self.input_dim)
        return dmdx, dvdx
     
    def predictive_gradient_mean(self, x):
            
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True
        
        # define the function we want to differentiate
        m = lambda x: self.model.posterior(x).mean.sum()
        
        with self.computation_context:
            dmdx  = torch.autograd.functional.jacobian(m, x)

        return self.scale * dmdx.numpy().reshape(-1, self.input_dim)

    def _resolve_bounds(self, bounds, parameter_names):

        if isinstance(bounds, dict):
            for param in parameter_names:
                if param not in bounds:
                    raise ValueError(f'Parameter \'{param}\' not found in bounds.')
            # convert to list
            return [bounds[param] for param in parameter_names]
        else:
            return bounds

    @property
    def X(self):
        """Return input evidence."""
        if self.model._has_transformed_inputs:
            return self.model._original_train_inputs.numpy()
        else:
            return self.model.train_inputs[0].numpy()

    @property
    def Y(self):
        """Return output evidence."""
        if hasattr(self.model, 'outcome_transform'):
            y = self.model.outcome_transform.untransform(self.model.train_targets)[0]
        else:
            y = self.model.train_targets
        return self.scale * y.numpy().reshape(-1,1)

    @property
    def instance(self):
        """Return the gp instance."""
        return self.model
