"""This module contains abstract base classes for active learning components."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

import elfi.visualization.interactive as visin
import elfi.visualization.visualization as vis


class ActiveLearner(ABC):
    """Base class for active learners used in parameter inference."""

    @abstractmethod
    def update(self, x, y, optimize=True):
        """Update model with new evidence.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize model fit.

        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_model(self):
        """Return current model fit or None if model has not been fitted.

        Returns
        -------
        PredictiveDistributionModel or None

        """
        pass

    @abstractmethod
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
        pass

    def plot_state(self, **options):
        """Plot the surrogate model and acquisition function.

        This feature is still experimental and currently supports only 2D cases.

        """
        f = plt.gcf()
        if len(f.axes) < 2:
            f, _ = plt.subplots(1, 2, figsize=(
                13, 6), sharex='row', sharey='row')

        gp = self.get_model()

        # Draw the GP surface
        visin.draw_contour(
            gp.predict_mean,
            gp.bounds,
            gp.parameter_names,
            title='GP target surface',
            points=gp.X,
            axes=f.axes[0],
            **options)

        # Draw the latest acquisitions
        if options.get('interactive'):
            point = gp.X[-1, :]
            if len(gp.X) > 1:
                f.axes[1].scatter(*point, color='red')

        displays = [gp.instance]

        if options.get('interactive'):
            from IPython import display
            displays.insert(
                0,
                display.HTML('<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                    len(gp.Y), gp.Y[-1][0], point)))

        # Update
        visin._update_interactive(displays, options)

        def acq(x):
            return self.evaluate_acquisition_function(x, t=len(gp.X))

        # Draw the acquisition surface
        visin.draw_contour(
            acq,
            gp.bounds,
            gp.parameter_names,
            title='Acquisition surface',
            points=None,
            axes=f.axes[1],
            **options)

        if options.get('close'):
            plt.close()


class PredictiveDistributionModel(ABC):
    """Base class for predictive distribution models."""

    @abstractmethod
    def predict(self, x):
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def plot(self, axes=None, **options):
        """Plot pairwise relationships as a matrix with inputs vs. targets.

        Returns
        -------
        axes : np.array of plt.Axes

        """
        return vis.plot_gp(self, self.parameter_names, **options)

    def plot_discrepancy(self, **options):
        """Plot observed inputs vs targets.

        Return
        ------
        axes : np.array of plt.Axes

        """
        return vis.plot_discrepancy(self, self.parameter_names, **options)

    @property
    @abstractmethod
    def X(self):
        """Return input evidence as numpy array."""
        pass

    @property
    @abstractmethod
    def Y(self):
        """Return output evidence as numpy array."""
        pass

    @property
    @abstractmethod
    def instance(self):
        """Return the model instance."""
        pass
