"""This module contains implementation of bolfire."""

__all__ = ['BOLFIRE']

import logging

import numpy as np

import elfi.methods.mcmc as mcmc
from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC, AcquisitionBase
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import CostFunction
from elfi.methods.classifier import Classifier, LogisticRegression
from elfi.methods.inference.parameter_inference import ModelBased
from elfi.methods.posteriors import BOLFIREPosterior
from elfi.methods.results import BOLFIRESample
from elfi.methods.utils import arr2d_to_batch, batch_to_arr2d, resolve_sigmas
from elfi.model.extensions import ModelPrior

logger = logging.getLogger(__name__)


class BOLFIRE(ModelBased):
    """Bayesian Optimization and Classification in Likelihood-Free Inference (BOLFIRE)."""

    def __init__(self,
                 model,
                 n_training_data,
                 feature_names=None,
                 classifier=None,
                 bounds=None,
                 n_initial_evidence=0,
                 acq_noise_var=0,
                 exploration_rate=10,
                 update_interval=1,
                 target_model=None,
                 acquisition_method=None,
                 batch_size=None,
                 **kwargs):
        """Initialize the BOLFIRE method.

        Parameters
        ----------
        model: ElfiModel
            Elfi graph used by the algorithm.
        n_training_data: int or list
            Size of training data.
        feature_names: str or list, optional
            ElfiModel nodes used as features in classification. Default all Summary nodes.
        classifier: str, optional
            Classifier to be used. Default LogisticRegression.
        bounds: dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters: dict('parameter_name': (lower, upper), ... ). Not used if
            custom target_model is given.
        n_initial_evidence: int, optional
            Number of initial evidence.
        acq_noise_var: float or dict, optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
            If a dictionary, values should be float specifying the variance for each dimension.
        exploration_rate: float, optional
            Exploration rate of the acquisition method.
        update_interval: int, optional
            How often to update the GP hyperparameters of the target_model.
        target_model: GPyRegression, optional
            A surrogate model to be used.
        acquisition_method: Acquisition, optional
            Method of acquiring evidence points. Default LCBSC.

        """
        batch_size = batch_size or 2 * np.min(n_training_data)
        assert batch_size % 2 == 0
        n_train = n_training_data if isinstance(n_training_data, int) else n_training_data[0]
        super(BOLFIRE, self).__init__(model, 2 * n_train, feature_names=feature_names,
                                      batch_size=batch_size, **kwargs)
        self._random_state = np.random.RandomState(self.seed)

        # Initialize classifier attributes
        self.classifier = self._resolve_classifier(classifier)

        # TODO: write resolvers for the attributes below
        self.bounds = bounds
        self.acq_noise_var = acq_noise_var
        self.exploration_rate = exploration_rate
        self.update_interval = update_interval

        # Initialize GP regression
        self.target_model = self._resolve_target_model(target_model)
        self.prior = ModelPrior(self.model, parameter_names=self.parameter_names)

        # Initialize BO
        self.n_initial_evidence = self._resolve_n_initial_evidence(n_initial_evidence)
        self.acquisition_method = self._resolve_acquisition_method(acquisition_method)

        # Adaptive simulation count
        self.is_multi = isinstance(n_training_data, list)
        self.current_index = 0
        if self.is_multi:
            self.n_batches_round = [int(2 * n_train/self.batch_size) for n_train in n_training_data]
            assert np.all(np.array(self.n_batches_round) >= 1)
            assert np.all(np.array(self.n_batches_round) <= self.n_batches_round[0])
        else:
            self.n_batches_round = [int(2 * n_training_data/self.batch_size)]

        # Initialize state dictionary
        self.state['n_evidence'] = 0
        self.state['last_GP_update'] = self.n_initial_evidence

        # Initialize classifier attributes list
        self.classifier_attributes = []

        # Initialize data collection
        self._init_round()

    @property
    def parameter_names(self):
        """Return the parameters to be inferred."""
        return self.target_model.parameter_names

    @property
    def n_evidence(self):
        """Return the number of acquired evidence points."""
        return self.state['n_evidence']

    @property
    def finished(self):
        """Check whether objective has been reached."""
        round_reached = self.objective['round'] <= self.state['round']
        batches_reached = self.objective['n_batches'] <= self.state['n_batches']
        return round_reached or batches_reached

    def set_objective(self, rounds, n_sim=None):
        """Set an objective for inference.

        Parameters
        ----------
        rounds: int
            Number of data collection rounds.
        n_sim: int, optional
            Number of simulations. Inference stops when either rounds or n_sim is reached.

        """
        self.objective['round'] = rounds
        if n_sim is None:
            self.objective['n_batches'] = rounds * int(self.n_sim_round / self.batch_size)
        else:
            self.objective['n_batches'] = int(n_sim / self.batch_size)

    def extract_result(self):
        """Extract the results from the current state."""
        return BOLFIREPosterior(self.parameter_names,
                                self.target_model,
                                self.prior,
                                self.classifier_attributes)

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index: int

        Returns
        -------
        batch: dict

        """
        params = np.atleast_2d(self.current_params)
        params_1 = np.repeat(params, int(self.batch_size/2), axis=0)
        params_0 = self.prior.rvs(int(self.batch_size/2), random_state=self._random_state)
        batch_params = np.vstack((params_1, params_0))
        return arr2d_to_batch(batch_params, self.parameter_names)
        
    def predict_log_ratio(self, X, y, X_obs):
        """Predict the log-ratio, i.e, logarithm of likelihood / marginal.

        Parameters
        ----------
        X: np.ndarray
            Training data features.
        y: np.ndarray
            Training data labels.
        X_obs: np.ndarray
            Observed data.

        Returns
        -------
        np.ndarray

        """
        self.classifier.fit(X, y)
        return self.classifier.predict_log_likelihood_ratio(X_obs)

    def fit(self, n_evidence, n_sim=None, bar=True):
        """Fit the surrogate model.

        That is, generate a regression model for the negative posterior value given the parameters.
        Currently only GP regression are supported as surrogate models.

        Parameters
        ----------
        n_evidence: int
            Number of evidence for fitting.
        n_sim: int, optional
            Number of simulations. Inference stops when either n_evidence or n_sim is reached.
        bar: bool, optional
            Flag to show or hide the progress bar during fit.

        Returns
        -------
        BOLFIREPosterior

        """
        logger.info('BOLFIRE: Fitting the surrogate model...')
        if isinstance(n_evidence, int) and n_evidence > 0:
            if n_evidence < self.n_evidence:
                logger.warning('Requesting less evidence than there already exists.')
            return self.infer(n_evidence, n_sim=n_sim, bar=bar)
        raise TypeError('n_evidence must be a positive integer.')

    def sample(self,
               n_samples,
               warmup=None,
               n_chains=4,
               initials=None,
               algorithm='nuts',
               sigma_proposals=None,
               n_evidence=None,
               *args, **kwargs):
        """Sample from the posterior distribution of BOLFIRE.

        Sampling is performed with an MCMC sampler.

        Parameters
        ----------
        n_samples: int
            Number of requested samples from the posterior for each chain. This includes warmup,
            and note that the effective sample size is usually considerably smaller.
        warmup: int, optional
            Length of warmup sequence in MCMC sampling.
        n_chains: int, optional
            Number of independent chains.
        initials: np.ndarray (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain.
        algorithm: str, optional
            Sampling algorithm to use.
        sigma_proposals: np.ndarray
            Standard deviations for Gaussian proposals of each parameter for Metropolis-Hastings.
        n_evidence: int, optional
            If the surrogate model is not fitted yet, specify the amount of evidence.

        Returns
        -------
        BOLFIRESample

        """
        # Fit posterior in case not done
        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        # Check algorithm
        if algorithm not in ['nuts', 'metropolis']:
            raise ValueError('The given algorithm is not supported.')

        # Check standard deviations of Gaussian proposals when using Metropolis-Hastings
        if algorithm == 'metropolis':
            sigma_proposals = resolve_sigmas(self.parameter_names,
                                             sigma_proposals,
                                             self.target_model.bounds)

        posterior = self.extract_result()
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with best likelihood ratio
        if initials is not None:
            if np.asarray(initials).shape != (n_chains, self.target_model.input_dim):
                raise ValueError('The shape of initials must be (n_chains, n_params).')
        else:
            inds = np.argsort(self.target_model.Y[:, 0])
            initials = np.asarray(self.target_model.X[inds])

        # Enable caching for default RBF kernel
        self.target_model.is_sampling = True

        tasks_ids = []
        ii_initial = 0
        for ii in range(n_chains):
            seed = get_sub_seed(self.seed, ii)
            # Discard bad initialization points
            while np.isinf(posterior.logpdf(initials[ii_initial])):
                ii_initial += 1
                if ii_initial == len(inds):
                    raise ValueError('BOLFIRE.sample: Cannot find enough acceptable '
                                     'initialization points!')

            if algorithm == 'nuts':
                tasks_ids.append(
                    self.client.apply(mcmc.nuts,
                                      n_samples,
                                      initials[ii_initial],
                                      posterior.logpdf,
                                      posterior.gradient_logpdf,
                                      n_adapt=warmup,
                                      seed=seed,
                                      **kwargs))

            elif algorithm == 'metropolis':
                tasks_ids.append(
                    self.client.apply(mcmc.metropolis,
                                      n_samples,
                                      initials[ii_initial],
                                      posterior.logpdf,
                                      sigma_proposals,
                                      warmup,
                                      seed=seed,
                                      **kwargs))

            ii_initial += 1

        # Get results from completed tasks or run sampling (client-specific)
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get_result(id))

        chains = np.asarray(chains)

        logger.info(f'{n_chains} chains of {n_samples} iterations acquired. '
                    'Effective sample size and Rhat for each parameter:')
        for ii, node in enumerate(self.parameter_names):
            logger.info(f'{node} {mcmc.eff_sample_size(chains[:, :, ii])} '
                        f'{mcmc.gelman_rubin_statistic(chains[:, :, ii])}')

        self.target_model.is_sampling = False

        return BOLFIRESample(method_name='BOLFIRE',
                             chains=chains,
                             parameter_names=self.parameter_names,
                             warmup=warmup,
                             n_sim=self.state['n_sim'],
                             seed=self.seed,
                             *args, **kwargs)

    def _resolve_classifier(self, classifier):
        """Resolve classifier."""
        if classifier is None:
            return LogisticRegression()
        if isinstance(classifier, Classifier):
            return classifier
        raise ValueError('classifier must be an instance of Classifier.')

    def _resolve_n_initial_evidence(self, n_initial_evidence):
        """Resolve number of initial evidence."""
        if isinstance(n_initial_evidence, int) and n_initial_evidence >= 0:
            return n_initial_evidence
        raise ValueError('n_initial_evidence must be a non-negative integer.')

    def _resolve_target_model(self, target_model):
        """Resolve target model."""
        if target_model is None:
            return GPyRegression(self.model.parameter_names, self.bounds)
        if isinstance(target_model, GPyRegression):
            return target_model
        raise TypeError('target_model must be an instance of GPyRegression.')

    def _resolve_acquisition_method(self, acquisition_method):
        """Resolve acquisition method."""
        if acquisition_method is None:
            # Model prior log-probabilities as an additive cost
            cost = CostFunction(self.prior.logpdf, self.prior.gradient_logpdf, scale=-1)
            return LCBSC(model=self.target_model,
                         prior=self.prior,
                         noise_var=self.acq_noise_var,
                         exploration_rate=self.exploration_rate,
                         seed=self.seed,
                         additive_cost=cost)
        if isinstance(acquisition_method, AcquisitionBase):
            return acquisition_method
        raise TypeError('acquisition_method must be an instance of AcquisitionBase.')

    @property
    def current_params(self):
        """Return parameter values explored in the current round."""
        return self._current_params

    @current_params.setter
    def current_params(self, params):
        """Set parameter values explored in the current round."""
        self._current_params = params

    def _init_round(self):
        """Initialise a new data collection round.

        BOLFIRE uses an acquisition method to choose parameter values.

        """
        self.state['n_sim_round'] = 0

        # Set new parameter values
        if self.n_evidence < self.n_initial_evidence:
            # Sample parameter values from the model priors
            self.current_params = self.prior.rvs(1, random_state=self._random_state)
        else:
            # Acquire parameter values from the acquisition function
            t = self.n_evidence - self.n_initial_evidence
            if self.is_multi:
                self.current_params, self.current_index = self.acquisition_method.acquire(1, t)
                self.n_sim_round = self.n_batches_round[self.current_index] * self.batch_size
            else:
                self.current_params = self.acquisition_method.acquire(1, t)

        # Maximum batch index that can be prepared
        self.batch_index_max = self.state['n_batches'] + self.n_batches_round[self.current_index]

    def _allow_submit(self, batch_index):
        """Check whether batch_index can be prepared."""
        # Do not prepare batches with new parameter values until the current round is finished
        if batch_index >= self.batch_index_max and self.batches.has_pending:
            return False
        else:
            return super(BOLFIRE, self)._allow_submit(batch_index)

    def _process_simulated(self):
        """Process the simulated data.

        BOLFIRE uses the simulated data to calculate log-ratio estimates and update the
        surrogate model.

        """
        if self.is_multi and self.n_evidence < self.n_initial_evidence:
            # Predict log-ratio at all fidelities
            negative_log_ratio_value = np.zeros(len(self.n_batches_round))
            for index, n_batches in enumerate(self.n_batches_round):
                X, y = self._generate_training_data(n_batches)
                negative_log_ratio_value[index] = -1 * self.predict_log_ratio(X, y, self.observed)
                self.classifier_attributes += [self.classifier.attributes]
        else:
            # Predict log-ratio
            X, y = self._generate_training_data(self.n_batches_round[self.current_index])
            negative_log_ratio_value = -1 * self.predict_log_ratio(X, y, self.observed)
            self.classifier_attributes += [self.classifier.attributes]

        # BO part

        self.state['n_evidence'] += 1
        optimize = self._should_optimize()
        parameter_values = self.current_params
        if self.is_multi:
            if self.n_evidence <= self.n_initial_evidence:
                params = np.repeat(parameter_values, len(self.n_batches_round), axis=0)
                inds = np.arange(len(self.n_batches_round))
            else:
                params = parameter_values
                inds = np.array([self.current_index])
            self.target_model.update(params, negative_log_ratio_value, inds, optimize)
        else:
            self.target_model.update(parameter_values, negative_log_ratio_value, optimize)
        if optimize:
            self.state['last_GP_update'] = self.state['n_evidence']

    def _generate_training_data(self, n_batches):
        """Generate training data."""
        n_training_data = int(n_batches * self.batch_size)
        X = self.simulated[:n_training_data]
        y = np.concatenate((np.ones(int(self.batch_size/2)), -1 * np.ones(int(self.batch_size/2))))
        y = np.tile(y, n_batches)
        return X, y

    def _should_optimize(self):
        """Check whether GP hyperparameters should be optimized."""
        current = self.state['n_evidence'] + 1
        next_update = self.state['last_GP_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update
