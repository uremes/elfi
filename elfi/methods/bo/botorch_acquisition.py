import torch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound

class BoTorchLCBSC(BoTorchAcquisition):
    
    def __init__(self,
                 model,
                 exploration_rate=10,
                 optim_params=None
                 ):
        """Initialize LCBSC acquisition method.

        Parameters
        ----------
        model : BoTorchModel
            Gaussian process regression model.
        exploration_rate : float, optional
            Must be exploration_rate > 0.
        optim_params : Dict[str, Any], optional
            Acquisition function optimisation parameters.

        """
        self.model = model
        self.input_dim = self.model.input_dim
        self.bounds = torch.tensor(np.transpose(self.model.bounds), dtype=torch.double)
        
        self.acq_class = UpperConfidenceBound
        self.exploration_rate = exploration_rate
        self.acq_options = {'beta': 1/self.exploration_rate, 'maximize': False}
        self.optim_params = optim_params or {'num_restarts': 10, 'raw_samples': 500}

    def evaluate(self, x, t=None):
        """Evaluate the acquisition function value at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
        t : int
            current acquisition index
        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """ 
        if t is not None:
            self.acq_options['beta'] = self.beta(t+1, self.input_dim, 1/self.exploration_rate)
        else:
            self.acq_options['beta'] = self.exploration_rate

        return super().evaluate(x)

    def acquire(self, n, t=None):
        """Return the next batch of acquisition points.

        Parameters
        ----------
        n : int
            number of acquisition points to return
        t : int
            current acquisition index
        Returns
        -------
        np.array
            with shape (n, input_dim)

        """
        if t is not None:
            self.acq_options['beta'] = self.beta(t+1, self.input_dim, 1/self.exploration_rate)
        else:
            self.acq_options['beta'] = self.exploration_rate

        return super().acquire(n)
    
    def beta(self, t, d, delta):
        """Calculate beta based on the update rule used in ELFI.

        Parameters
        ----------
        t : int
            current acquisition index
        d : int
            input dimension
        delta : float
            Must be delta > 0 and recommended delta < 1.
        """
        return 2 * np.log(t**(2 * d + 2) * np.pi**2 / (3 * delta))
