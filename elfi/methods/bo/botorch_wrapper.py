"""This module contains wrappers for using BoTorch in ELFI."""

import contextlib

import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var

from elfi.methods.bo.acquisition import AcquisitionBase
from elfi.methods.bo.gpy_regression import GPyRegression


class BoTorchModel(GPyRegression):

    def __init__(self,
                 parameter_names,
                 bounds,
                 model_class=None,
                 model_options=None,
                 mll_class=None,
                 mll_options=None,
                 negate=False,
                 use_fast_pred_var=True,
                 seed=None):
        """Initialize BoTorch model wrapper.

        Parameters
        ----------
        parameter_names : List[str]
            Input parameter names.
        bounds : Dict[str, Sequence[float, float]].
            Lower and upper bound for each input parameter.
        model_class : Type[botorch.models.Model], optional
            Model type.
        model_options : Dict[str, Any], optional
            model_class constructor parameters
        mll_class : Type[gpytorch.mlls.MarginalLogLikelihood], optional
            Model fit type.
        mll_options : Dict[str, Any], optional
            mll_class constructor parameters
        negate : bool, optional
            If True, negate target values.
        use_fast_pred_var : bool, optional
            If True, use fast predictive variance computation.
        seed : int, optional

        """
        self.parameter_names = parameter_names
        self.input_dim = len(self.parameter_names)
        self.bounds = [bounds[param] for param in parameter_names]
        self.model_class = model_class or SingleTaskGP
        self.model_options = model_options or {}
        self.mll_class = mll_class or ExactMarginalLogLikelihood
        self.mll_options = mll_options or {}
        self.sign = 1 if not negate else -1
        if fast_pred_var:
            self.computation_context = fast_pred_var()
        else:
            self.computation_context = contextlib.nullcontext()

        self.train_x = []
        self.train_y = []
        self.model = None

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

        # activate evaluation mode
        self.model.eval()
        self.model.likelihood.eval()

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

        # activate evaluation mode
        self.model.eval()

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

        # activate evaluation mode
        self.model.eval()

        # define the function we want to differentiate
        def m(x):
            return self.model.posterior(x).mean.sum()

        with self.computation_context:
            dmdx = torch.autograd.functional.jacobian(m, x)

        return self.sign * dmdx.numpy().reshape(-1, self.input_dim)

    def update(self, x, y, optimize=True):
        """Update model with new evidence.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize model fit.

        """
        y = self.sign * y
        self.train_x.append(x)
        self.train_y.append(y)

        if self.model is not None:
            self._init_model(state_dict=self.model.state_dict())

        if optimize:
            self.optimize()

    def optimize(self):
        """Optimize model fit."""

        if self.model is None:
            self._init_model()

        mll = self.mll_class(self.model.likelihood, self.model, **self.mll_options)
        fit_gpytorch_model(mll)

    def _init_model(self, state_dict=None):

        x = torch.tensor(np.array(self.train_x), dtype=torch.double).reshape(-1, self.input_dim)
        y = torch.tensor(np.array(self.train_y), dtype=torch.double).reshape(-1, 1)
        self.model = self.model_class(x, y, **self.model_options)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    @property
    def X(self):
        return np.array(self.train_x).reshape(-1, self.input_dim)

    @property
    def Y(self):
        return self.sign * np.array(self.train_y).reshape(-1, 1)

    @property
    def n_evidence(self):
        return len(self.train_x)

    @property
    def instance(self):
        return self.model

    # because bayesian optimisation in ELFI assumes that a target model has attribute _gp and that it can access it
    @property
    def _gp(self):
        return self.model

class BoTorchAcquisition(AcquisitionBase):

    def __init__(self,
                 model,
                 acq_class,
                 acq_options,
                 optim_params=None
                 ):
        """Initialize BoTorch acquisition method.

        Parameters
        ----------
        model : BoTorchModel
            Gaussian process regression model.
        bounds : Dict[str, Sequence[float, float]] or List[Sequence[float, float]]
            Lower and upper bound for each input parameter.
        acq_class : Type[botorch.acquisition.AcquisitionFunction]
            Acquisition function type.
        acq_options : Dict[str, Any], optional
            acq_class constructor parameters
        optim_params : Dict[str, Any], optional
            Acquisition function optimisation parameters.

        """
        self.model = model
        self.input_dim = self.model.input_dim
        self.bounds = torch.tensor(np.transpose(self.model.bounds), dtype=torch.double)
        
        self.acq_class = acq_class
        self.acq_options = acq_options
        self.optim_params = optim_params or {'num_restarts': 5, 'raw_samples': 20}

    def evaluate(self, x, t=None):
        """Evaluate the acquisition function value at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
        t : int
            current acquisition index (unused)

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """
        if self.model.instance is None:
            return np.zeros((x.shape[0], 1))

        x = torch.tensor(x, dtype=torch.double).reshape(-1, 1, self.input_dim)
        acq_function = self.acq_class(self.model.instance, **self.acq_options)

        return acq_function(x).detach().numpy()

    def acquire(self, n, t=None):
        """Return the next batch of acquisition points.

        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acquisition index (unused).

        Returns
        -------
        np.array
            with shape (n, input_dim)

        """
        if self.model.instance is None:
            self.model.optimize()

        acq_function = self.acq_class(self.model.instance, **self.acq_options)
        x, _ = optimize_acqf(acq_function, bounds=self.bounds, q=n, **self.optim_params)

        return x.numpy()
