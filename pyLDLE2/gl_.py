import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

def graph_laplacian(neigh_dist, neigh_ind, k_nn, k_tune, gl_type,
                    return_diag=False, use_out_degree=True,
                    tune_type=0):
    if type(k_tune) != list:
        assert k_nn > k_tune, "k_nn must be greater than k_tune."
    assert gl_type in ['symnorm','unnorm', 'diffusion'],\
            "gl_type should be one of {'symnorm','unnorm','diffusion'}"
    
    n = neigh_dist.shape[0]
    
    if type(k_tune) == list: # epsilon ball based graph laplacian
        epsilon = neigh_dist[:,k_nn-1]
        p, d = k_tune
        autotune = 4*0.5*((epsilon**2)/chi2.ppf(p, df=d))
        autotune = autotune[:,np.newaxis]
    else:
        # Compute tuning values for each pair of neighbors
        sigma = neigh_dist[:,k_tune-1].flatten()
        if tune_type == 0: # scaling depends on sigma_i and sigma_j
            autotune = sigma[neigh_ind]*sigma[:,np.newaxis]
        elif tune_type == 1: # scaling depends on sigma_i only
            autotune = np.repeat(sigma**2, neigh_ind.shape[1])
            autotune = autotune.reshape(neigh_ind.shape)
        elif tune_type == 2: # scaling is fixed across data points
            autotune = np.median(sigma)**2
    
    # Compute kernel matrix
    if tune_type == 3: # Binary kernel no tuning
        K = np.ones(neigh_dist.shape)
    else:
        eps = np.finfo(np.float64).eps
        K = np.exp(-neigh_dist**2/(autotune+eps))
    
    # Convert to sparse matrices
    source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
    K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    
    
    # symmetrize
    K = K + K.T
    ones_K_like = ones_K_like + ones_K_like.T
    K.data /= ones_K_like.data
    #K = K + K.T - K.multiply(K.T)
    
    if gl_type == 'diffusion':
        Dinv = 1/(K.sum(axis=1).reshape((n,1)))
        K = K.multiply(Dinv).multiply(Dinv.transpose())
        gl_type = 'symnorm'

    # Compute and return graph Laplacian based on gl_type
    if gl_type == 'symnorm':
        return autotune,\
               laplacian(K, normed=True,
                         return_diag=return_diag,
                         use_out_degree=use_out_degree)
    elif gl_type == 'unnorm':
        return autotune,\
               laplacian(K, normed=False,
                         return_diag=return_diag,
                         use_out_degree=use_out_degree)

class GL:
    def __init__(self, debug=False):
        self.debug = debug
        self.autotune = None
        self.L = None
        self.v0 = None
        self.lmbda0 = None
        self.phi0 = None
        self.lmbda = None
        self.phi = None
    
    def fit(self, neigh_dist, neigh_ind, local_opts):
        # Eigendecomposition of graph Laplacian
        # Note: Eigenvalues are returned sorted.
        # Following is needed for reproducibility of lmbda and phi
        np.random.seed(42)
        v0 = np.random.uniform(0,1,neigh_dist.shape[0])
        gl_type = local_opts['gl_type']
        
        if gl_type in ['unnorm', 'symnorm']:
            autotune, L = graph_laplacian(neigh_dist, neigh_ind, local_opts['k_nn'],
                                           local_opts['k_tune'], gl_type)
            lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, which='SM')
            # TODO: Why the following doesn't give correct eigenvalues
            # lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, sigma=0.0)
        else:
            if gl_type == 'random_walk':
                gl_type = 'symnorm'
            autotune, L_and_sqrt_D = graph_laplacian(neigh_dist, neigh_ind, local_opts['k_nn'],
                                            local_opts['k_tune'], gl_type,
                                            return_diag = True)
            L, sqrt_D = L_and_sqrt_D
            lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, which='SM')
            # TODO: Why the following doesn't give correct eigenvalues
            # lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, sigma=0.0)
            
            L = L.multiply(1/sqrt_D[:,np.newaxis]).multiply(sqrt_D[np.newaxis,:])
            phi = phi/sqrt_D[:,np.newaxis]
            
            # TODO: Is this normalization needed?
            phi = phi/(np.linalg.norm(phi,axis=0)[np.newaxis,:])

        if self.debug:
            # The trivial eigenvalue and eigenvector
            self.lmbda0 = lmbda[0]
            self.phi0 = phi[:,0][:,np.newaxis]
            self.L = L
            self.v0 = v0
            self.autotune = autotune
        
        # Remaining eigenvalues and eigenvectors
        self.lmbda = lmbda[1:]
        self.phi = phi[:,1:]