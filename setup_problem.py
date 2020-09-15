from utils.prob import Problem
import numpy as np

problem = Problem()
A = np.load("../data/matrices/128x128sequency_hadamard_matrix.npy")
problem.build_prob(A)
problem.save("./experiments/m10_n100_k0.0_p0.1_s0/prob")
