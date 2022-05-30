"""This module contains wrappers for using BoTorch in ELFI."""

import contextlib
import copy

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
        self._gp = None

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

        if self._gp is None:
            return (np.zeros(x.shape[0], 1), np.ones(x.shape[0], 1))

        # activate evaluation mode
        self._gp.eval()
        self._gp.likelihood.eval()

        with torch.no_grad(), self.computation_context:
            pred = self._gp.posterior(x, observation_noise=not(noiseless))

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

        if self._gp is None:
            return (np.zeros(x.shape[0], self.input_dim), np.zeros(x.shape[0], self.input_dim))

        # activate evaluation mode
        self._gp.eval()

        # define the mean and variance function that we want to differentiate
        def m(x):
            return self._gp.posterior(x).mean.sum()

        def v(x):
            return self._gp.posterior(x).variance.sum()

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

        if self._gp is None:
            return np.zeros(x.shape[0], self.input_dim)

        # activate evaluation mode
        self._gp.eval()

        # define the function we want to differentiate
        def m(x):
            return self._gp.posterior(x).mean.sum()

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
        xt = torch.tensor(np.array(self.train_x), dtype=torch.double).reshape(-1, self.input_dim)
        yt = torch.tensor(np.array(self.train_y), dtype=torch.double).reshape(-1, 1)

        if self._gp is None:
            # initialise
            self._gp = self._make_model_instance(xt, yt)
        else:
            # reconstruct with new data
            self._gp = self._make_model_instance(xt, yt, state_dict=self._gp.state_dict())

        if optimize:
            self.optimize()

    def optimize(self):
        """Optimize model fit."""
        if self._gp is None:
            raise RuntimeError('Model has not been initialised.')

        mll = self.mll_class(self._gp.likelihood, self._gp, **self.mll_options)
        fit_gpytorch_model(mll)

    def _make_model_instance(self, x, y, state_dict=None):
        model = self.model_class(x, y, **self.model_options)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        return len(self.train_x)

    @property
    def X(self):
        """Return input evidence."""
        return np.array(self.train_x).reshape(-1, self.input_dim)

    @property
    def Y(self):
        """Return output evidence."""
        return self.sign * np.array(self.train_y).reshape(-1, 1)

    @property
    def noise(self):
        """Return the noise."""
        if self._gp is None:
            return None
        else:
            return self._gp.likelihood.noise.detach().numpy()

    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    def copy(self):
        """Return a copy of current instance."""
        return copy.deepcopy(self)


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
            raise RuntimeError('Model has not been initialised.')

        acq_function = self.acq_class(self.model.instance, **self.acq_options)
        x, _ = optimize_acqf(acq_function, bounds=self.bounds, q=n, **self.optim_params)

        return x.numpy()
