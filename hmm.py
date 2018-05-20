'Module for hidden Markov models'

from collections import deque
from io import StringIO
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
        """
        Markov state transition matrix

        A (N, N) right stochastic matrix giving the probabilities of state
        transition: state_transition[i,j] is the probability that system state
        will transition from state i to state j.
        """
        return self._Phi

    @property
    def emission(self):
        """
        Emission Matrix

        A (N, M) right stochastic matrix giving the conditional probability of
        the observations: emission[i, j] is the probability of observing
        observation j conditioned on state i.
        """
        return self._Theta

    @property
    def n_state_space(self):
        'Cardinality of the state space (N)'
        return self._N

    @property
    def n_obs_space(self):
        'Cardinality of the Observation space (M)'
        return self._M

    def __iter__(self):
        'infinte Iterator of (state, observation)'
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
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
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

class ViterbiDecoder:
    """
    Online Viterbi decoder

    The online Viterbi decoder tracks the minimum cost path in a finite-state
    trellis structure. It does this by tracking survivor segments and minimum
    total state costs. At each time index, the total cost of the cheapest path
    through the trellis to each state is updated, as well as a cumulative list
    of survivor segments for each of the state nodes.

    This is often used to track the Maximum a posteriori Probability estimate
    of a discrete-time, finite-state Markov process observed in memoryless
    noise.

    The main way a ViterbiDecoder instance is used, is by iteratively updating
    the survivor segment list by calling update() with the current time's
    transition cost matrix. At any time, the Viterbi paths for each state can
    be determined by backtracing the survivor segment list.
    """

    def __init__(self, costs):
        """
        Initialize the decoder with a set of initial state node costs. The
        state space dimension of the decoder is determined at this time by the
        length of `costs` and remains constant.

        Parameters
        ----------
        costs : array (N,) float
            initial costs of each state.
        """
        self._N = len(costs)
        self._survivors = deque([np.arange(self._N)])
        self._costs = costs

    def update(self, A):
        """
        Update survivor segment list and node costs using a transition cost
        matrix. This is the main function of the ViterbiDecoder. Clients
        iteratively update the state of the ViterbiDecoder by calling this for
        each time index. This propagates the current time forward by one index.

        Parameters
        ----------
        A : array (N, N) float
            Cost matrix for current state transition. A[i,j] gives the cost of
            going from state i at the current time index to state j in the next
            time index.
        """
        X = (self._costs + A.T).T
        self._survivors += [np.argmin(X, 0)]
        self._costs = np.min(X, 0)

    def traceback(self):
        """
        Return an iterator of back-traced survivor paths. Yields an np.ndarray
        of shape (N,) where for yielded value x, x[i]'s traverse the Viterbi
        path for leading to state i backward from the current time to the root
        node.
        """
        y = range(self._N)
        for s in reversed(self._survivors):
            y = s[y]
            yield y

    def trace(self):
        """
        Return an iterator tracing the Viterbi paths from the current root node
        to each state node at the current time index. Equivalent to
        reversed(list(self.traceback())). See traceback() documentation for
        details.
        """
        return reversed(list(self.traceback()))

    def prune(self):
        """
        Return the longest common Viterbi path set (as a deque) and prune it
        from survivors list. This frees memory used to store survivor segments
        which future path decisions cannot depend. This re-roots the trellis at
        the most recent time index on which all current Viterbi paths traverse
        the same state node.
        """
        commonpath = deque()
        for e in self.trace():
            if np.all(e == e[0]):
                self._survivors.popleft() # munch
                commonpath += [e[0]]
        return commonpath

    def __str__(self):
        """
        Print a trace of the survivor transitions and the Viterbi paths from
        the current root node to the current time index. Time indices run down
        lines of the output. Survivor transitions are on the left of the "|"
        and Viterbi paths are on the right.
        """
        sio = StringIO()
        for s, t in zip(self._survivors, self.trace()):
            print("".join(f"{n:d} " for n in s), "| ",
                  "".join(f"{n:d} " for n in t), sep="", file=sio)
        return sio.getvalue()
