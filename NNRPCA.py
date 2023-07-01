import numpy as np
import numpy.linalg as LA

def NRPCA(D, opts):
    # Extract options
    max_iter = opts.get('max_iter', 100)        # Maximum number of iterations
    tol = opts.get('tol', 1e-6)                  # Tolerance for convergence
    lambda_L = opts.get('lambda_L', 1 / np.sqrt(np.max(D.shape)))    # Regularization parameter for low-rank component
    lambda_S = opts.get('lambda_S', 1 / np.sqrt(np.max(D.shape)))    # Regularization parameter for sparse component
    mu = opts.get('mu', 1.25 / LA.norm(D, 2))    # Penalty parameter
    rho = opts.get('rho', 1.6)                   # Parameter for updating mu
    
    # Initialize variables
    m, n = D.shape
    L = np.zeros((m, n))                         # Low-rank component
    S = np.zeros((m, n))                         # Sparse component
    U_S = np.zeros((m, n))                       # Auxiliary variable for S
    V_S = np.zeros((n, n))                       # Auxiliary variable for S
    count_L_final = 0                            # Counter for final low-rank convergence
    count_S_final = 0                            # Counter for final sparse convergence
    sparsity_S = 0                               # Sparsity of the sparse component
    error_D_final = 0                            # Final reconstruction error of D
    error_S_final = 0                            # Final reconstruction error of S
    k = 0                                        # Unused variable (set to 0)
    iter = 0                                     # Current iteration count
    
    # Singular Value Thresholding function
    def svd_thresholding(X, tau):
        U, s, V = LA.svd(X, full_matrices=False)
        return U @ np.diag(np.maximum(s - tau, 0)) @ V
    
    while iter < max_iter:
        L_prev = L
        S_prev = S
        U_S_prev = U_S
        V_S_prev = V_S
        
        # Update L using low-rank approximation
        L = svd_thresholding(D - S + U_S / mu, lambda_L / mu)
        L = np.maximum(L, 0)                     # Nonnegative constraint on L
        
        # Update S using soft-thresholding
        S = np.maximum(D - L + U_S / mu - lambda_S / mu, 0) - np.maximum(-(D - L + U_S / mu) + lambda_S / mu, 0)
        
        # Update U_S and V_S
        U_S = U_S + mu * (D - L - S)
        V_S = V_S_prev + mu * (S - S_prev)
        
        # Update mu
        mu = min(mu * rho, 1e10)
        
        # Compute convergence criteria
        err_L = LA.norm(L - L_prev, 'fro') / LA.norm(L_prev, 'fro')     # Relative change in L
        err_S = LA.norm(S - S_prev, 'fro') / LA.norm(S_prev, 'fro')     # Relative change in S
        err_D = LA.norm(D - L - S, 'fro') / LA.norm(D, 'fro')           # Relative reconstruction error of D
        
        # Update iteration count
        iter += 1
        
        # Check convergence
        if err_L < tol and err_S < tol and err_D < tol:
            count_L_final += 1
            count_S_final += 1
            error_D_final = err_D
            error_S_final = err_S
            
            sparsity_S = np.sum(S != 0) / (m * n)    # Compute sparsity of S
            
            if iter >= 2:
                if count_L_final >= 2 and count_S_final >= 2:
                    break
            
    return L, S, U_S, V_S, count_L_final, count_S_final, sparsity_S, error_D_final, error_S_final, k, iter
