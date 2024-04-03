import numpy as np
from scipy.linalg import svd

def nonneg_rpca(Y, lambda_val=1.0, mu=1.0, max_iter=100, rho=1.6):
    
    m, n = Y.shape

    L = np.zeros((m, n))  # Initialize low-rank matrix
    S = np.zeros((m, n))  # Initialize sparse matrix
    E = np.zeros((m, n))  # Initialize auxiliary variable

    for iter in range(max_iter):
        # Update low-rank component (L) using Singular Value Thresholding (SVT)
        U, Sigma, Vt = svd(Y - S + (1 / rho) * E, full_matrices=False)
        shrinkage = np.maximum(Sigma - 1.0 / rho, 0)
        L = np.dot(U, np.dot(np.diag(shrinkage), Vt))

        # Update sparse component (S) with soft thresholding
        S = np.maximum(Y - L + (1 / rho) * E - lambda_val / rho, 0)

        # Update auxiliary variable (E)
        E = E + rho * (Y - L - S)

        # Apply nonnegative constraint with penalty mu
        L = np.maximum(L - mu, 0)

    return L, S
