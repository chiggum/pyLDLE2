import pdb
import time
import numpy as np
import itertools
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import floyd_warshall, shortest_path
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
import multiprocess as mp
import os
import pickle

def path_exists(path):
    return os.path.exists(path) or os.path.islink(path)

def makedirs(dirpath):
    if path_exists(dirpath):
        return
    os.makedirs(dirpath)

def read(fpath, verbose=True):
    if not path_exists(fpath):
        if verbose:
            print(fpath, 'does not exist.')
        return None
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    if verbose:
        print('Read data from', fpath, flush=True)
    return data
    
def save(dirpath, fname, data, verbose=True):
    if not path_exists(dirpath):
        os.makedirs(dirpath)
    fpath = dirpath + '/' + fname
    with open(fpath, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print('Saved data in', fpath, flush=True)
        
        
def shortest_paths(X, n_nbrs):
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(X)
    knn_graph = nbrs.kneighbors_graph(mode='distance')
    dist_matrix, predecessors = shortest_path(knn_graph, return_predecessors=True, directed=False)
    return dist_matrix, predecessors

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
                 algo = 'LPCA',
                 **kwargs):
        self.algo = algo
        self.T = None
        self.v = None
        self.b = None
        # Following variables are
        # initialized externally
        # i.e. by the caller
        self.zeta = None
        self.noise_seed = None
        self.noise_var = 0
        
        # For LDLE
        self.Psi_gamma = None
        self.Psi_i = None
        self.phi = None
        
        # For LPCA and its variants
        self.Psi = None
        self.mu = None
        self.X = None
        self.y = None
        
        # For KPCA, ISOMAP etc
        self.model = None
        self.X = None
        self.y = None
        
        self.add_dim = False
        
    def eval_(self, opts):
        k = opts['view_index']
        mask = opts['data_mask']
        
        if self.algo == 'LDLE':
            temp = self.Psi_gamma[k,:][np.newaxis,:]*self.phi[np.ix_(mask,self.Psi_i[k,:])]
            n = self.phi.shape[0]
        elif self.algo == 'LPCA':
            temp = np.dot(self.X[mask,:]-self.mu[k,:][np.newaxis,:],self.Psi[k,:,:])
            n = self.X.shape[0]
        else:
            temp = self.model[k].transform(self.X[mask,:])
        
        if self.noise_var:
            np.random.seed(self.noise_seed[k])
            temp2 = np.random.normal(0, self.noise_var, (n, temp.shape[1]))
            temp = temp + temp2[mask,:]
            
        if self.add_dim:
            temp = np.concatenate([temp,np.zeros((temp.shape[0],1))], axis=1)
        
        if self.b is None:
            return temp
        else:
            temp = self.b[k]*temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k,:,:])
            if self.v is not None:
                temp = temp + self.v[[k],:]
            return temp
        
    def reconstruct_(self, opts):
        k = opts['view_index']
        y_ = opts['embeddings']
        if self.algo == 'LDLE':
            pass
        elif self.algo == 'LPCA':
            temp = np.dot(np.dot(y_-self.v[[k],:], self.T[k,:,:].T),self.Psi[k,:,:].T)+self.mu[k,:][np.newaxis,:]
        else:
            pass
        return temp
    
    def out_of_sample_eval_(self, opts):
        k = opts['view_index']
        X_ = opts['out_of_samples']
        
        if self.algo == 'LDLE':
            temp = self.Psi_gamma[k,:][np.newaxis,:]*self.phi[np.ix_(mask,self.Psi_i[k,:])]
            n = self.phi.shape[0]
        elif self.algo == 'LPCA':
            temp = np.dot(X_-self.mu[k,:][np.newaxis,:],self.Psi[k,:,:])
            n = self.X.shape[0]
        else:
            temp = self.isomap[k].transform(X_)
        
        if self.noise_var:
            np.random.seed(self.noise_seed[k])
            temp2 = np.random.normal(0, self.noise_var, (n, temp.shape[1]))
            temp = temp + temp2[mask,:]
            
        if self.add_dim:
            temp = np.concatenate([temp,np.zeros((temp.shape[0],1))], axis=1)
        
        if self.b is None:
            return temp
        else:
            temp = self.b[k]*temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k,:,:])
            if self.v is not None:
                temp = temp + self.v[[k],:]
            return temp
    
    def alignment_wts(self, opts):
        beta = opts['beta']
        if beta is None:
            return None
        k = opts['view_index']
        mask = opts['data_mask']
        mu = np.mean(self.X[mask,:], axis=0)
        temp = self.X[mask,:] - mu[None,:]
        w = -np.linalg.norm(temp, 1, axis=1)/beta
        return w
        #p = np.exp(w - np.max(w))
        #p *= (temp.shape[0]/np.sum(p))
        #return p
    def repulsion_wts(self, opts):
        beta = opts['beta']
        if beta is None:
            return None
        k = opts['pt_index']
        far_off_pts = opts['repelling_pts_indices']
        if self.y is not None:
            temp = self.y[far_off_pts,:] - self.y[k,:][None,:]
            w = np.linalg.norm(temp, 2, axis=1)**2
            #temp0 = self.X[far_off_pts,:] - self.X[k,:][None,:]
            #w0 = np.linalg.norm(temp0, 2, axis=1)**2
            #p = 1.0*((w-w0)<0)
            p = 1/(w + 1e-12)
        else:
            p = np.ones(len(far_off_pts))
        return p


# includes self as the first neighbor
# data is either X or distance matrix d_e
def nearest_neighbors(data, k_nn, metric, n_jobs=-1, sort_results=True):
    n = data.shape[0]
    if k_nn > 1:
        neigh = NearestNeighbors(n_neighbors=k_nn-1, metric=metric, n_jobs=n_jobs)
        neigh.fit(data)
        neigh_dist, neigh_ind = neigh.kneighbors()
        neigh_dist = np.insert(neigh_dist, 0, np.zeros(n), axis=1)
        neigh_ind = np.insert(neigh_ind, 0, np.arange(n), axis=1)
        if sort_results:
            inds = np.argsort(neigh_dist, axis=-1)
            for i in range(neigh_ind.shape[0]):
                neigh_ind[i,:] = neigh_ind[i,inds[i,:]]
                neigh_dist[i,:] = neigh_dist[i,inds[i,:]]
    else:
        neigh_dist = np.zeros((n,1))
        neigh_ind = np.arange(n).reshape((n,1)).astype('int')
    return neigh_dist, neigh_ind
            
def sparse_matrix(neigh_ind, neigh_dist):
    if neigh_ind.dtype == 'object':
        row_inds = []
        col_inds = []
        data = []
        for k in range(neigh_ind.shape[0]):
            row_inds.append(np.repeat(k, neigh_ind[k].shape[0]).tolist())
            col_inds.append(neigh_ind[k].tolist())
            data.append(neigh_dist[k].tolist())
        row_inds = list(itertools.chain.from_iterable(row_inds))
        col_inds = list(itertools.chain.from_iterable(col_inds))
        data = list(itertools.chain.from_iterable(data))
    else:
        row_inds = np.repeat(np.arange(neigh_dist.shape[0]), neigh_dist.shape[1])
        col_inds = neigh_ind.flatten()
        data = neigh_dist.flatten()
    return csr_matrix((data, (row_inds, col_inds)))

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

def ixmax(x, k=0, idx=None):
    col = x[idx, k] if idx is not None else x[:, k]
    z = np.where(col == col.max())[0]
    return z if idx is None else idx[z]

def lexargmax(x):
    idx = None
    for k in range(x.shape[1]):
        idx = ixmax(x, k, idx)
        if len(idx) < 2:
            break
    return idx[0]

def compute_distortion_at(y_d_e, s_d_e):
    scale_factors = (y_d_e+1e-12)/(s_d_e+1e-12)
    mask = np.ones(scale_factors.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_distortion = np.max(scale_factors[mask])/np.min(scale_factors[mask])
    print('Max distortion is:', max_distortion, flush=True)
    n = y_d_e.shape[0]
    distortion_at = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = 0
        distortion_at[i] = np.max(scale_factors[i,mask])/np.min(scale_factors[i,mask])
        mask[i] = 1
    return distortion_at, max_distortion

def compute_prctile_distortion_at(y_d_e, s_d_e, prctile=50):
    scale_factors = (y_d_e+1e-12)/(s_d_e+1e-12)
    np.fill_diagonal(scale_factors,1)
    max_distortion = np.percentile(scale_factors, prctile)/np.percentile(scale_factors, 100-prctile)
    print('Max distortion is:', max_distortion, flush=True)
    n = y_d_e.shape[0]
    distortion_at = np.zeros(n)
    for i in range(n):
        distortion_at[i] = np.percentile(scale_factors[i,:], prctile)/np.percentile(scale_factors[i,:], 100-prctile)
    return distortion_at, max_distortion

def get_path_lengths_in_embedding_space(s_d_e, pred, y_d_e, n_proc=8, verbose=True):
    n = pred.shape[0]
    inds = np.arange(n)
    y_d_e2 = np.zeros((n,n))
    
    def target_proc(pairs_to_proc, start_ind, end_ind, q_, y_d_e, s_d_e, pred):
        def get_path_length(i, j):
            path_length = 0
            k = j
            while pred[i, k] != -9999:
                path_length += y_d_e[k, pred[i, k]]
                k = pred[i, k]
            return path_length

        my_data = np.zeros(end_ind-start_ind)
        for ind in range(start_ind, end_ind):
            i,j = pairs_to_proc[ind]
            my_data[ind-start_ind] = get_path_length(i,j)

        q_.put((start_ind, end_ind, my_data))

    pairs_to_proc = list(itertools.combinations(np.arange(n), 2))
    q_ = mp.Queue()
    chunk_sz = len(pairs_to_proc)//n_proc
    proc = []
    start_ind = 0
    end_ind = 1
    for p_num in range(n_proc):
        if p_num == n_proc-1:
            end_ind = len(pairs_to_proc)
        else:
            end_ind = (p_num+1)*chunk_sz

        proc.append(mp.Process(target=target_proc,
                               args=(pairs_to_proc, start_ind, end_ind, q_, y_d_e, s_d_e, pred),
                               daemon=True))
        proc[-1].start()
        start_ind = end_ind

    print('All processes started', flush=True)
    for p_num in range(n_proc):
        start_ind, end_ind, y_d_e2_ = q_.get()
        for ind in range(start_ind, end_ind):
            i,j = pairs_to_proc[ind]
            y_d_e2[i,j] = y_d_e2_[ind-start_ind]
            y_d_e2[j,i] = y_d_e2[i,j]

    q_.close()
    for p_num in range(n_proc):
        proc[p_num].join()
    
    return y_d_e2

def reconstruct_(self, opts):
    k = opts['view_index']
    y_ = opts['embeddings']
    if self.algo == 'LDLE':
        return None
    elif self.algo == 'LPCA':
        temp = np.dot(np.dot(y_-self.v[[k],:], self.T[k,:,:].T),self.Psi[k,:,:].T)+self.mu[k,:][np.newaxis,:]
    elif self.algo == 'LISOMAP':
        return None
    return temp

def reconstruct_data(p, buml_obj, y, is_init=False, averaging=True):
    if averaging:
        Utildeg = buml_obj.GlobalViews.compute_Utildeg(y, buml_obj.IntermedViews.C, buml_obj.global_opts)
        Utilde = buml_obj.IntermedViews.Utilde.multiply(Utildeg)
        Utilde.eliminate_zeros() 
    else:
        Utilde = buml_obj.IntermedViews.C
    if is_init:
        intermed_param = buml_obj.GlobalViews.intermed_param_init
    else:
        intermed_param = buml_obj.IntermedViews.intermed_param
    m,n = Utilde.shape
    X = np.zeros((n, p))
    for k in range(n):
        views = Utilde[:,k].nonzero()[0].tolist()
        temp = []
        embedding = y[k,:][None,:]
        for i in views:
            temp.append(reconstruct_(intermed_param, {'view_index': i, 'embeddings': embedding}))
        temp = np.array(temp)
        X[k,:] = np.mean(temp, axis=0)
    return X
    

def compute_global_distortions(X, y, n_nbr=10, buml_obj_path='',
                               read_dir_root='', save_dir_root='',
                               n_proc=32):
    # Shortest paths in the data
    s_d_e_path = read_dir_root + '/s_d_e.dat'
    pred_path = read_dir_root + '/pred.dat'
    save0 = (read_dir_root != '')
    if (not path_exists(s_d_e_path)) or (not path_exists(pred_path)):
        s_d_e, pred = shortest_paths(X, n_nbr)
        if save0:
            save(read_dir_root, 's_d_e.dat', s_d_e)
            save(read_dir_root, 'pred.dat', pred)
    else:
        s_d_e = read(s_d_e_path)
        pred = read(pred_path)
    
    # Shortest paths in the embedding
    save1 = (save_dir_root != '')
    y_s_d_e_path = save_dir_root + '/y_s_d_e.dat'
    if (not path_exists(y_s_d_e_path)):
        y_s_d_e, _ = shortest_paths(y, n_nbr)

        if (buml_obj_path is not None):
            all_data = read(buml_obj_path)
            X, labelsMat, buml_obj, gv_info, ex_name = all_data
            if buml_obj.global_opts['to_tear']:
                intermed_param = buml_obj.IntermedViews.intermed_param
                Utilde = buml_obj.IntermedViews.Utilde
                C = buml_obj.IntermedViews.C
                global_opts = buml_obj.global_opts
                y_s_d_e = gv_info['gv'].compute_pwise_dist_in_embedding(intermed_param,
                                                                        Utilde, C, global_opts,
                                                                        dist=y_s_d_e, y=y)
        if save1:
            save(save_dir_root, 'y_s_d_e.dat', y_s_d_e)
    else:
        y_s_d_e = read(y_s_d_e_path)
        
    # Lengths of the embeddings of the shortest paths in the data
    save2 = (save_dir_root != '')
    y_d_e2_path = save_dir_root + '/y_d_e2.dat'
    if not path_exists(y_d_e2_path):
        y_d_e2 = get_path_lengths_in_embedding_space(s_d_e, pred, y_s_d_e,
                                                     n_proc=n_proc, verbose=True)
        if save2:
            save(save_dir_root, 'y_d_e2.dat', y_d_e2)
    else:
        y_d_e2 = read(y_d_e2_path)
    
    sd_at, max_sd = compute_distortion_at(y_s_d_e, s_d_e)
    wd_at, max_wd = compute_distortion_at(y_d_e2, s_d_e)
    return sd_at, max_sd, wd_at, max_wd