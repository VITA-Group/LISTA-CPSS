from utils.prob import Problem
import numpy as np

problem = Problem()
A = np.load("../data/matrices/500x500_gaussian_matrix.npy")
problem.build_prob(A)
problem.save("./experiments/m250_n500_k0.0_p0.1_sinf/prob")
