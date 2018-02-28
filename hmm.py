#pylint: disable=invalid-name

'Module for hidden Markov models'

import numpy as np

class HiddenMarkovModel:

    'Hidden Markov Model.'

    def __init__(self, state_transition, emission, initial_state):
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
    def N(self):
        'Cardinality of the state space'
        return self._N

    @property
    def M(self):
        'Cardinality of the Observation space'
        return self._M

    def __iter__(self):
        while True:
            self._x = np.random.choice(self._N, p=self._Phi[self._x, :])
            y = np.random.choice(self._M, p=self._Theta[self._x, :])
            yield self._x, y
