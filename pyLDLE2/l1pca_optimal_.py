"""
Optimal L1-PCA Algorithm with time complexity O[N^(DK-K+1)]
Markopoulos, P.P., Karystinos, G.N. and Pados, D.A., 2014. Optimal algorithms for $ L_{1} $-subspace signal processing. IEEE Transactions on Signal Processing, 62(19), pp.5046-5058. doi: 10.1109/TSP.2014.2338077.

Input: data matrix (X) and number of PCs to be found (K)
Output: optimal binary matrix (B), projection matrix (Q) and L1-projection metric ||Q.T@X||_L1
"""

# modified by Dhruv Kohli

from numpy.linalg import svd
from scipy.sparse.linalg import svds
import numpy as np
import itertools
import math

def mysign(x, tol = 1e-7):
    x[np.abs(x) < tol] = 0
    return np.sign(x)

def decimal2binary(decimal, numberOfBits):
    N = len(decimal)
    B = np.zeros((numberOfBits, N))
    for n in range(N):
        B[:, n] = np.array(list(np.binary_repr(n).zfill(numberOfBits))).astype(np.int8)
    return 2*B-1

def compute_candidates(matrix, halfSphere = True):
    rho = matrix.shape[0]
    N = matrix.shape[1]
    assert rho <= N, 'Input matrix should have full-row rank.'
    numOfambiguities = rho-1
    Bpool = decimal2binary(list(range(2**numOfambiguities)), numOfambiguities)
    combinations = list(itertools.combinations((range(N)), numOfambiguities))
    candidates = set()
    for combination in combinations:
        matrixI = matrix[:, combination]
        factorU, diagS, factorVt = np.linalg.svd(matrixI, full_matrices = True)
        v = factorU[:, -1]
        b_ambiguous = mysign(matrix.T@v).flatten()
        for n in range(Bpool.shape[1]):
            b = b_ambiguous.copy()
            try:
                b[b == 0] = Bpool[:, n].copy()
            except:
                print('Check tolerance of mysign function.')
                exit()
            b = tuple(b[0]*b)
            candidates.add(b)
    if not halfSphere:
        otherHalf = set()
        for b in candidates:
            otherHalf.add(tuple(-np.array(b)))
        candidates = candidates.union(otherHalf)
    return np.array(list(candidates)).T

def l1pca_optimal(X, K):
    D = X.shape[0]
    N = X.shape[1]
    U, S, V = svds(X.T, k=K, random_state=42)
    Q = U @ np.diag(S)
    B = compute_candidates(Q.T)
    P = B.shape[1]
    Z = list(itertools.combinations(range(P), K))
    metopt = 0
    for i in range(len(Z)):
        met = np.linalg.norm(X @ B[:, Z[i]], 'nuc')
        if met > metopt:
            metopt = met
            zopt = Z[i]
    Bopt = B[:,zopt]
    U, _, V = svd(X@Bopt, full_matrices=False)
    Qopt = U@V.T
    return Qopt, Bopt, metopt