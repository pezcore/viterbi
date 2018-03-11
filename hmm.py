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
