'Module for hidden Markov models'

import numpy as np

def rand_right_stoch(size):
    'Return a random uniformly distributed right stochastic matrix'
    a = np.random.rand(*size)
    return (a.T / a.sum(1)).T

class HiddenMarkovModel:

    'Hidden Markov Model.'

    def __init__(self, state_transition, emission, initial_state=0):
        """
        initialize the HiddenMarkovModel.

        Parameters
        ----------
        state_transition : array (N, N)
            state transition probability matrix. state_transition[i][j] is the
            probability of transition from state i to state j
        emission : array (N, M)
            measurement model emission matrix. emission[i][j] is the
            probability of observing observation j from state i.
        initial_state : int
        """

        self._x = initial_state
        self._Phi = state_transition
        self._Theta = emission
        self._M = emission.shape[1]
        shp = state_transition.shape
        self._N = shp[0]
        assert len(shp) == 2 and shp[0] == shp[1]

    @property
    def state(self):
        'Markov chain state (hidden)'
        return self._x

    @property
    def state_transition(self):
        'Markov state transition matrix'
        return self._Phi

    @property
    def emission(self):
        'Emission Matrix'
        return self._Theta

    @property
    def n_state_space(self):
        'Cardinality of the state space'
        return self._N

    @property
    def n_obs_space(self):
        'Cardinality of the Observation space'
        return self._M

    def __iter__(self):
        while True:
            self._x = np.random.choice(self._N, p=self._Phi[self._x, :])
            y = np.random.choice(self._M, p=self._Theta[self._x, :])
            yield self._x, y

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. This is a right stochastic matrix giving the
        probabilities of state transition: A[i,j] is the probability that
        system state will transition from state i to state j.
    B :  array (K, M)
        Emission matrix. A right stochastic matrix giving the conditional
        probability of the observations: B[i, j] is the probability of
        observing observation j conditioned on state i.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2
