"This module contains wrappers for using BoTorch in ELFI."

import copy
import contextlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.acquisition import AnalyticAcquisitionFunction, UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

class BoTorchWrapper():

    def __init__(self,
                 parameter_names,
                 bounds,
                 likelihood=None,
                 covar_module=None,
                 acq_method=None,
                 acq_params=None,
                 optim_params=None,
                 ):

        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = self._resolve_bounds(bounds, self.parameter_names)
        self.likelihood = likelihood
        self.covar_module = covar_module
        self.acq_method = acq_method or LCBSC
        self.acq_params = acq_params or {'exploration_rate': 10}
        self.optim_params = optim_params or {'num_restarts': 5, 'raw_samples': 20}

        # TODO add device

        self.init_x = []
        self.init_y = []

        self.model = None

    def update(self, x, y, optimize=True):

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

        if n > 1 and issubclass(self.acq_method, AnalyticAcquisitionFunction):
            raise ValueError('Selected acquisition method does not work with batch size > 1.')

        # update acquisition index
        self.acq_params['t'] = t

        acq_function = self.acq_method(self.model, **self.acq_params)
        x, value = optimize_acqf(acq_function, bounds=self.bounds, q=n, **self.optim_params)

        return x.numpy()

    def get_model(self):

        if self.model is not None:
            bounds = list(np.transpose(self.bounds.numpy()))
            return RegressionModel(copy.deepcopy(self.model), self.parameter_names, bounds)
        else:
            return None

    def get_acquisition_value(self, x, t=None):

        x = torch.tensor(x, dtype=torch.double).reshape(-1, 1, self.input_dim)

        # update acquisition index
        self.acq_params['t'] = t

        acq_function = self.acq_method(self.model, **self.acq_params)
        return acq_function(x).detach().numpy()

    def get_model_pred(self, x):

        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)

        # activate evaluation mode
        self.model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.model(x)

        return pred.mean.numpy()

    def get_evidence(self):

        return self.train_x.numpy(), self.train_y.numpy()

    def _resolve_bounds(self, bounds, parameter_names):

        # check that all parameters have bounds
        for param in parameter_names:
            if param not in bounds:
                raise ValueError(f'Parameter \'{param}\' not found in bounds.')
        # convert bounds to botorch format
        bounds = [bounds[param] for param in parameter_names]
        return torch.tensor(np.transpose(bounds), dtype=torch.double)

    def _init_model(self, x, y, state_dict=None):
        
        self.model = SingleTaskGP(x, y, likelihood=self.likelihood, covar_module=self.covar_module)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def _optimize(self):

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)


class LCBSC(UpperConfidenceBound):
    """
    Lower confidence bound selection criterion as implemented in ELFI.
    """

    def __init__(self, model, exploration_rate, t, **kwargs):

        beta = self._beta(1/exploration_rate, model.train_inputs[0].shape[1], t)
        super().__init__(model, beta=beta, maximize=False)
        
    def _beta(self, delta, d, t):
        # Start from 0
        t += 1
        return 2 * np.log(t**(2 * d + 2) * np.pi**2 / (3 * delta))


class RegressionModel():

    def __init__(self, model, parameter_names, bounds, fast_pred_var=True):

        self.model = model
        self.parameter_names = parameter_names
        self.bounds = bounds
        self.input_dim = len(bounds)
        if fast_pred_var:
            self.computation_context = gpytorch.settings.fast_pred_var()
        else:
            self.computation_context = contextlib.nullcontext()

        self.model.eval()
        self.model.likelihood.eval()

    def predict(self, x, noiseless=False):
        
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)

        with torch.no_grad(), self.computation_context:
            pred = self.model(x)
            if not noiseless: pred = self.model.likelihood(pred)

        return pred.mean.numpy().reshape(-1, 1), pred.variance.detach().numpy().reshape(-1, 1)
    
    def predict_mean(self, x):
        
        return self.predict(x, noiseless=True)[0]
    
    def predictive_gradients(self, x):
        
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True
        
        # define the mean and variance function that we want to differentiate
        m = lambda x: self.model(x).mean.sum()
        v = lambda x: self.model(x).variance.sum()
        
        with self.computation_context:
            dmdx  = torch.autograd.functional.jacobian(m, x)
            dvdx  = torch.autograd.functional.jacobian(v, x)
        
        return dmdx.numpy().reshape(-1, self.input_dim), dvdx.numpy().reshape(-1, self.input_dim)
     
    def predictive_gradient_mean(self, x):
            
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True
        
        # define the function we want to differentiate
        m = lambda x: self.model(x).mean.sum()
        
        with self.computation_context:
            dmdx  = torch.autograd.functional.jacobian(m, x)

        return dmdx.numpy().reshape(-1, self.input_dim)

    def _resolve_bounds(self, bounds, parameter_names):
        # TODO check that all param names are in bounds and raise informative errors
        # convert dict to list
        if isinstance(bounds, dict):
            return [bounds[param] for param in parameter_names]
        else:
            return bounds

    @property
    def X(self):
        """Return input evidence."""
        return self.model.train_inputs[0].numpy()

    @property
    def Y(self):
        """Return output evidence."""
        return self.model.train_targets.numpy().reshape(-1,1)

    @property
    def instance(self):
        """Return the gp instance."""
        return self.model
