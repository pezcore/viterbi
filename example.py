import numpy as np
from hmm import HiddenMarkovModel, ViterbiDecoder, rand_right_stoch

A = rand_right_stoch((4, 4))
B = rand_right_stoch((4, 6))
PI = rand_right_stoch((1, 4)).squeeze()

x0 = np.random.choice(4, p=PI)
hmm = HiddenMarkovModel(x0)
z0 = np.random.choice(6, p=B[hmm.state])
λ0 = -np.log(PI) - np.log(B[:, z0])

vd = ViterbiDecoder(λ0)

x = np.empty(50, "B")
z = np.empty(50, "B")

for i in range(50):
    x[i], z[i] = hmm.next(A, B)
    λ = -np.log(A) - np.log(B[:, z[i]]) # I think this is right
    vd.update(λ)
