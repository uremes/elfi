"""This module contains BoTorch implementations for ELFI acquisition functions."""

import numpy as np
from botorch.acquisition import UpperConfidenceBound


class LCBSC(UpperConfidenceBound):
    """Lower confidence bound selection criterion as implemented in ELFI.

    Does not support batch acquisitions.
    """

    def __init__(self, model, t, delta=0.1):
        """Initialize acquisition function.

        Parameters
        ----------
        model : botorch.models.model.Model
            A fitted single-outcome GP model.
        t : int
            Current acquisition index (starting from 0).
        delta : float, optional
            Represents exploitation tendency. Must be delta > 0 and recommended delta < 1.

        """
        # calculate beta based on the update rule used in ELFI
        t += 1
        d = model.train_inputs[0].shape[1]
        beta = 2 * np.log(t**(2 * d + 2) * np.pi**2 / (3 * delta))

        # initialise acquisition function
        super().__init__(model, beta=beta, maximize=False)
