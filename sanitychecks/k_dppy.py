import numpy as np
import os
from dppy.finite_dpps import FiniteDPP


r, N = 5, 10
# Random feature vectors
corr_mat = np.array([[0, 10, 1, 20, 2],
                [10, 0, 1, 15, 1],
                [1,  1, 0, 1, 10],
                [20,15, 1, 0,  1],
                [2, 1, 10, 1, 0]])
corr_mat = corr_mat @ corr_mat.T 
print(corr_mat)
DPP = FiniteDPP('likelihood', **{'L': corr_mat})

for _ in range(10):
    DPP.sample_exact_k_dpp(size=3)

print(DPP.list_of_samples)