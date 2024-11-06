import pdb
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from umap.umap_ import smooth_knn_dist, compute_membership_strengths

def umap_kernel(knn_indices, knn_dists, k_tune):
    n = knn_indices.shape[0]
    sigmas, rhos = smooth_knn_dist(knn_dists, k_tune, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    return result

# def sinkhorn(K, maxiter=10000, delta=1e-12, eps=0):
#     """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
#     D = np.array(K.sum(axis=1)).squeeze()
#     d0 = 1./(D+eps)
#     d1 = 1./(K.dot(d0)+eps)
#     d2 = 1./(K.dot(d1)+eps)
#     for tau in range(maxiter):
#         if np.max(np.abs(d0 / d2 - 1)) < delta:
#             print('Sinkhorn converged at iter:', tau)
#             break
#         d3 = 1. / (K.dot(d2) + eps)
#         d0=d1.copy()
#         d1=d2.copy()
#         d2=d3.copy()
#     d = np.sqrt(d2 * d1)
#     K.data = K.data*d[K.row]*d[K.col]
#     return K

def sinkhorn(K, maxiter=10000, delta=1e-12, eps=0, boundC = 1e-8, print_freq=1000):
    """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
    n = K.shape[0]
    r = np.ones((n,1))
    u = np.ones((n,1))
    v = r/(K.dot(u))
    x = np.sqrt(u*v)

    assert np.min(x) > boundC, 'assert min(x) > boundC failed.'
    for tau in range(maxiter):
        error =  np.max(np.abs(u*(K.dot(v)) - r))
        if tau%print_freq:
            print('Error:', error, flush=True)
        
        if error < delta:
            print('Sinkhorn converged at iter:', tau)
            break

        u = r/(K.dot(v))
        v = r/(K.dot(u))
        x = np.sqrt(u*v)
        if np.sum(x<boundC) > 0:
            print('boundC not satisfied at iter:', tau)
            x[x < boundC] = boundC
        
        u=x
        v=x
    x = x.flatten()
    K.data = K.data*x[K.row]*x[K.col]
    return K

def graph_laplacian(neigh_dist, neigh_ind, k_nn, k_tune, gl_type,
                    return_diag=False, use_out_degree=True,
                    tuning='self', doubly_stochastic_max_iter=0):
    if type(k_tune) != list:
        assert k_nn > k_tune, "k_nn must be greater than k_tune."
    assert gl_type in ['symnorm','unnorm', 'diffusion'],\
            "gl_type should be one of {'symnorm','unnorm','diffusion'}"
    
    n = neigh_dist.shape[0]
    
    if tuning == 'umap':
        K = umap_kernel(neigh_ind, neigh_dist, k_tune)
    else:
        if type(k_tune) == list: # epsilon ball based graph laplacian
            epsilon = neigh_dist[:,k_nn-1]
            p, d = k_tune
            autotune = 4*0.5*((epsilon**2)/chi2.ppf(p, df=d))
            autotune = autotune[:,np.newaxis]
        else:
            if tuning is not None:
                # Compute tuning values for each pair of neighbors
                sigma = neigh_dist[:,k_tune-1].flatten()
                if 'self' in tuning: # scaling depends on sigma_i and sigma_j
                    autotune = sigma[neigh_ind]*sigma[:,np.newaxis]
                elif 'solo' in tuning: # scaling depends on sigma_i only
                    autotune = np.repeat(sigma**2, neigh_ind.shape[1])
                    autotune = autotune.reshape(neigh_ind.shape)
                elif 'median' in tuning: # scaling is fixed across data points
                    autotune = np.median(sigma)**2

        eps = np.finfo(np.float64).eps
        # Compute kernel matrix
        if tuning is None: # Binary kernel no tuning
            K = np.ones(neigh_dist.shape)
            autotune = None
        elif 'laplacian' in tuning:
            K = np.exp(-neigh_dist/(np.sqrt(autotune)+eps))
        else:
            K = np.exp(-neigh_dist**2/(autotune+eps))+eps
    
        # Convert to sparse matrices
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten()+eps,(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    
    
        # symmetrize
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        #K = K + K.T - K.multiply(K.T)
        
        if doubly_stochastic_max_iter:
            K = sinkhorn(K.tocoo(), maxiter=doubly_stochastic_max_iter)
        
    
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
        #v0 = np.random.uniform(0,1,neigh_dist.shape[0])
        v0 = np.ones(neigh_dist.shape[0])/np.sqrt(neigh_dist.shape[0])
        gl_type = local_opts['gl_type']
        tuning = local_opts['tuning']
        
        if gl_type in ['unnorm', 'symnorm']:
            autotune, L = graph_laplacian(neigh_dist, neigh_ind, local_opts['k_nn'],
                                           local_opts['k_tune'], gl_type, tuning=tuning,
                                            doubly_stochastic_max_iter=local_opts['doubly_stochastic_max_iter'])
            #lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, which='SM')
            # TODO: Why the following doesn't give correct eigenvalues
            lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, sigma=-1e-3)
        else:
            if gl_type == 'random_walk':
                gl_type = 'symnorm'
            autotune, L_and_sqrt_D = graph_laplacian(neigh_dist, neigh_ind, local_opts['k_nn'],
                                            local_opts['k_tune'], gl_type,
                                            return_diag = True, tuning=tuning,
                                            doubly_stochastic_max_iter=local_opts['doubly_stochastic_max_iter'])
            L, sqrt_D = L_and_sqrt_D
            #lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, which='SM')
            # TODO: Why the following doesn't give correct eigenvalues
            lmbda, phi = eigsh(L, k=local_opts['N']+1, v0=v0, sigma=-1e-3)
            
            L = L.multiply(1/sqrt_D[:,np.newaxis]).multiply(sqrt_D[np.newaxis,:])
            phi = phi/sqrt_D[:,np.newaxis]
            
            # TODO: Is this normalization needed?
            phi = phi/(np.linalg.norm(phi,axis=0)[np.newaxis,:])

        if self.debug:
            # The trivial eigenvalue and eigenvector
            self.lmbda0 = lmbda[0]
            self.phi0 = phi[:,0][:,np.newaxis]
            self.v0 = v0
            self.autotune = autotune
            #self.sqrt_D = sqrt_D
        
        # Remaining eigenvalues and eigenvectors
        self.L = L
        self.lmbda = lmbda[1:]
        self.phi = phi[:,1:]