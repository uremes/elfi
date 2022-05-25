"""This module contains wrappers for using BoTorch in ELFI."""

import contextlib
import copy

import numpy as np
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var

from elfi.methods.bo.active_learner import ActiveLearner, PredictiveDistributionModel
from elfi.methods.bo.botorch_acquisition import LCBSC


class BoTorchWrapper(ActiveLearner):
    """Active learner that uses BoTorch models and acquisition functions."""

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
                 negate=False,
                 seed=None
                 ):
        """Initialize active learner.

        Parameters
        ----------
        parameter_names : List[str]
            Input parameter names. Input dimension is inferred as len(parameter_names).
        bounds : Dict[str, Sequence[float, float]]
            Lower and upper bound for each input parameter.
        model_class : Type[botorch.models.Model], optional
            Model type.
        model_options : Dict[str, Any], optional
            model_class constructor parameters
        mll_class : Type[gpytorch.mlls.MarginalLogLikelihood], optional
            Model fit type.
        mll_options : Dict[str, Any], optional
            mll_class constructor parameters
        acq_class : Type[botorch.acquisition.AcquisitionFunction], optional
            Acquisition function type.
        acq_options : Dict[str, Any], optional
            acq_class constructor parameters
        optim_params : Dict[str, Any], optional
            Acquisition function optimisation parameters.
        negate : bool, optional
            If True, negate target function.
        seed : int, optional

        """
        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = self._resolve_bounds(bounds, self.parameter_names)
        self.model_class = model_class or SingleTaskGP
        self.model_options = model_options or self._get_default_options(self.bounds)
        self.mll_class = mll_class or ExactMarginalLogLikelihood
        self.mll_options = mll_options or {}
        self.acq_class = acq_class or LCBSC
        self.acq_options = acq_options or {'delta': 0.1}
        self.optim_params = optim_params or {'num_restarts': 5, 'raw_samples': 20}
        self.negate = negate
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.init_x = []
        self.init_y = []
        self.model = None

    def update(self, x, y, optimize=True):
        """Update model with new evidence.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize model fit.

        """
        if self.negate:
            y = -y

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
        if n > 1 and issubclass(self.acq_class, AnalyticAcquisitionFunction):
            raise ValueError('Selected acquisition class does not support batch size > 1.')

        # update acquisition index
        if self.acq_class == LCBSC:
            self.acq_options['t'] = t

        acq_function = self.acq_class(self.model, **self.acq_options)
        x, value = optimize_acqf(acq_function, bounds=self.bounds, q=n, **self.optim_params)

        return x.numpy()

    def get_model(self):
        """Return current model fit.

        Returns
        -------
        BoTorchModel or None

        """
        if self.model is not None:
            model = copy.deepcopy(self.model)
            bounds = list(np.transpose(self.bounds.numpy()))
            return BoTorchModel(model, self.parameter_names, bounds, negate=self.negate)
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
        if self.model is None:
            return np.zeros((x.shape[0], 1))

        x = torch.tensor(x, dtype=torch.double).reshape(-1, 1, self.input_dim)

        # update acquisition index
        if self.acq_class == LCBSC:
            self.acq_options['t'] = t

        acq_function = self.acq_class(self.model, **self.acq_options)
        return acq_function(x).detach().numpy()

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
            raise TypeError('bounds must be a dictionary.')

    def _get_default_options(self, bounds):

        options = {}
        # use the active learner bounds to normalise inputs
        if not(all(bounds[0] == 0) and all(bounds[1] == 1)):
            options['input_transform'] = Normalize(bounds.shape[1], bounds=bounds)
        # standardise outcome mean and variance
        options['outcome_transform'] = Standardize(1)

        return options

    def _init_model(self, x, y, state_dict=None):

        self.model = self.model_class(x, y, **self.model_options)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def _optimize(self):

        mll = self.mll_class(self.model.likelihood, self.model, **self.mll_options)
        fit_gpytorch_model(mll)


class BoTorchModel(PredictiveDistributionModel):
    """Predictive distribution model that uses a BoTorch model."""

    def __init__(self, model, parameter_names, bounds, negate=False, use_fast_pred_var=True):
        """Initialize predictive model.

        Parameters
        ----------
        model : botorch.models.Model
            Fitted Gaussian process model.
        parameter_names : List[str]
            Input parameter names.
        bounds : Dict[str, Sequence[float, float]] or List[Sequence[float, float]]
            Lower and upper bound for each input parameter.
        negate : bool, optional
            If True, negate target values.
        use_fast_pred_var : bool, optional
            If True, use fast predictive variance approximation.

        """
        self.model = model
        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = self._resolve_bounds(bounds, self.parameter_names)
        self.sign = 1 if not negate else -1
        if fast_pred_var:
            self.computation_context = fast_pred_var()
        else:
            self.computation_context = contextlib.nullcontext()

        self.model.eval()
        self.model.likelihood.eval()

    def predict(self, x, noiseless=False):
        """Return the model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        tuple
            model (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)

        """
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)

        with torch.no_grad(), self.computation_context:
            pred = self.model.posterior(x, observation_noise=not(noiseless))

        m = self.sign * pred.mean.detach().numpy().reshape(-1, 1)
        v = pred.variance.detach().numpy().reshape(-1, 1)
        return m, v

    def predict_mean(self, x):
        """Return the model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        return self.predict(x, noiseless=True)[0]

    def predictive_gradients(self, x):
        """Return the gradients of the model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        tuple
            model (grad_mean, grad_var) at x where
                grad_mean : np.array
                    with shape (x.shape[0], input_dim)
                grad_var : np.array
                    with shape (x.shape[0], input_dim)

        """
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True

        # define the mean and variance function that we want to differentiate
        def m(x):
            return self.model.posterior(x).mean.sum()

        def v(x):
            return self.model.posterior(x).variance.sum()

        with self.computation_context:
            dmdx = torch.autograd.functional.jacobian(m, x)
            dvdx = torch.autograd.functional.jacobian(v, x)

        dmdx = self.sign * dmdx.numpy().reshape(-1, self.input_dim)
        dvdx = dvdx.numpy().reshape(-1, self.input_dim)
        return dmdx, dvdx

    def predictive_gradient_mean(self, x):
        """Return the gradient of the model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """
        x = torch.tensor(x, dtype=torch.double).reshape(-1, self.input_dim)
        x.requires_grad = True

        # define the function we want to differentiate
        def m(x):
            return self.model.posterior(x).mean.sum()

        with self.computation_context:
            dmdx = torch.autograd.functional.jacobian(m, x)

        return self.sign * dmdx.numpy().reshape(-1, self.input_dim)

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
        return self.sign * y.numpy().reshape(-1, 1)

    @property
    def instance(self):
        """Return the model instance."""
        return self.model
