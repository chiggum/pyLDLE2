import pdb
import time
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import floyd_warshall
from sklearn.neighbors import NearestNeighbors

def print_log(s, log_time, local_start_time, global_start_time):
    print(s)
    if log_time:
        print('##############################')
        print('Time elapsed from last time log: %0.1f seconds' %(time.time()-local_start_time))
        print('Total time elapsed: %0.1f seconds' %(time.time()-global_start_time))
        print('##############################')
    return time.time()

class Param:
    def __init__(self,
                 algo = 'LDLE',
                 **kwargs):
        self.algo = algo
        self.T = None
        self.v = None
        self.b = None
        # Following variables are
        # initialized externally
        # i.e. by the caller
        self.zeta = None
        self.noise = 0
        
        # For LDLE
        self.Psi_gamma = None
        self.Psi_i = None
        self.phi = None
        
        # For LTSA
        self.Psi = None
        self.mu = None
        self.X = None
        
    def eval_(self, opts):
        k = opts['view_index']
        mask = opts['data_mask']
        
        if self.algo == 'LDLE':
            temp = self.Psi_gamma[k,:][np.newaxis,:]*self.phi[np.ix_(mask,self.Psi_i[k,:])]
        elif self.algo == 'LTSA':
            temp = np.dot(self.X[mask,:]-self.mu[k,:][np.newaxis,:],self.Psi[k,:,:])
        
        if self.noise:
            np.random.seed(k)
            temp2 = np.random.normal(0, self.noise, (mask.shape[0], temp.shape[1]))
            temp = temp + temp2[mask,:]
        
        if self.b is None:
            return temp
        else:
            temp = self.b[k]*temp
            if self.T is not None and self.v is not None:
                return np.dot(temp, self.T[k,:,:]) + self.v[[k],:]
            else:
                return temp


# includes self as the first neighbor
# data is either X or distance matric d_e
def nearest_neighbors(data, k_nn, metric, n_jobs=-1):
    neigh = NearestNeighbors(n_neighbors=k_nn-1, metric=metric, n_jobs=n_jobs)
    neigh.fit(data)
    neigh_dist, neigh_ind = neigh.kneighbors()
    n = neigh_dist.shape[0]
    neigh_dist = np.insert(neigh_dist, 0, np.zeros(n), axis=1)
    neigh_ind = np.insert(neigh_ind, 0, np.arange(n), axis=1)
    return neigh_dist, neigh_ind
            
def sparse_matrix(neigh_ind, neigh_dist):
    row_inds = np.repeat(np.arange(neigh_dist.shape[0]), neigh_dist.shape[1])
    col_inds = neigh_ind.flatten()
    return csr_matrix((neigh_dist.flatten(), (row_inds, col_inds)))

def to_dense(x):
    if issparse(x):
        return x.toarray()
    else:
        return x

def compute_zeta(d_e_mask0, Psi_k_mask):
    d_e_mask = to_dense(d_e_mask0)
    if d_e_mask.shape[0]==1:
        return 1
    d_e_mask_ = squareform(d_e_mask)
    mask = d_e_mask_!=0
    d_e_mask_ = d_e_mask_[mask]
    disc_lip_const = pdist(Psi_k_mask)[mask]/d_e_mask_
    return np.max(disc_lip_const)/(np.min(disc_lip_const) + 1e-12)


def custom_procrustes(X, Y, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - np.dot(muY, T)
   
    return T, c

# Solves for T, v s.t. T, v = argmin_{R,w)||AR + w - B||_F^2
# Here A and B have same shape n x d, T is d x d and v is 1 x d
def procrustes(A, B):
    T, c = custom_procrustes(B,A)
    return T, c