import pdb
import time
import numpy as np
import copy

from .util_ import print_log, compute_zeta
from .util_ import Param

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

import multiprocess as mp
from multiprocess import shared_memory
import itertools

# Computes cost_k, d_k (dest_k)
def cost_of_moving(k, d_e, neigh_ind_k, U_k, local_param, c, n_C,
                   Utilde, eta_min, eta_max):
    c_k = c[k]
    # Compute |C_{c_k}|
    n_C_c_k = n_C[c_k]
    
    # Check if |C_{c_k}| < eta_{min}
    # If not then c_k is already
    # an intermediate cluster
    if n_C_c_k >= eta_min:
        return np.inf, -1
    
    # Compute neighboring clusters c_{U_k} of x_k
    c_U_k = c[neigh_ind_k]
    c_U_k_uniq = np.unique(c_U_k).tolist()
    cost_x_k_to = np.zeros(len(c_U_k_uniq)) + np.inf
    
    # Iterate over all m in c_{U_k}
    i = 0
    for m in c_U_k_uniq:
        if m == c_k:
            i += 1
            continue
            
        # Compute |C_{m}|
        n_C_m = n_C[m]
        # Check if |C_{m}| < eta_{max}. If not
        # then mth cluster has reached the max
        # allowed size of the cluster. Move on.
        if n_C_m >= eta_max:
            i += 1
            continue
        
        # Check if |C_{m}| >= |C_{c_k}|. If yes, then
        # mth cluster satisfies all required conditions
        # and is a candidate cluster to move x_k in.
        if n_C_m >= n_C_c_k:
            # Compute union of Utilde_m U_k
            U_k_U_Utilde_m = list(U_k.union(Utilde[m]))
            # Compute the cost of moving x_k to mth cluster,
            # that is cost_{x_k \rightarrow m}
            cost_x_k_to[i] = compute_zeta(d_e[np.ix_(U_k_U_Utilde_m,U_k_U_Utilde_m)],
                                  local_param.eval_({'view_index': m,
                                                     'data_mask': U_k_U_Utilde_m}))
        i += 1
        
    
    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
    else:
        dest_k = c_U_k_uniq[dest_k]
        
    return cost_k, dest_k

class IntermedViews:
    def __init__(self, exit_at, verbose=True, debug=False):
        self.exit_at = exit_at
        self.verbose = verbose
        self.debug = debug
        
        self.c = None
        self.C = None
        self.n_C = None
        self.Utilde = None
        self.intermed_param = None
        
        self.local_start_time = time.time()
        self.global_start_time = time.time()
    
    def log(self, s='', log_time=False):
        if self.verbose:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    def fit(self, d, d_e, U, neigh_ind_, local_param, intermed_opts):
        n = d_e.shape[0]
        c = np.arange(n)
        n_C = np.zeros(n) + 1
        Clstr = list(map(set, np.arange(n).reshape((n,1)).tolist()))
        indices = U.indices
        indptr = U.indptr
        Utilde = []
        U_ = []
        neigh_ind = []
        for i in range(n):
            col_inds = indices[indptr[i]:indptr[i+1]]
            Utilde.append(set(col_inds))
            U_.append(set(col_inds))
            neigh_ind.append(col_inds)
        
        neigh_ind = np.array(neigh_ind)
        eta_max = intermed_opts['eta_max']
        n_proc = intermed_opts['n_proc']
        
        cost = np.zeros(n)+np.inf
        dest = np.zeros(n,dtype='int')-1
        
        shm_cost = shared_memory.SharedMemory(create=True, size=cost.nbytes)
        np_cost = np.ndarray(cost.shape, dtype=cost.dtype, buffer=shm_cost.buf)
        np_cost[:] = cost[:]
        shm_dest = shared_memory.SharedMemory(create=True, size=dest.nbytes)
        np_dest = np.ndarray(dest.shape, dtype=dest.dtype, buffer=shm_dest.buf)
        np_dest[:] = dest[:]
        
        shm_cost_name = shm_cost.name
        cost_shape = cost.shape
        cost_dtype = cost.dtype
        shm_dest_name = shm_dest.name
        dest_shape = dest.shape
        dest_dtype = dest.dtype
        
        
        # Vary eta from 2 to eta_{min}
        self.log('Constructing intermediate views.')
        for eta in range(2,intermed_opts['eta_min']+1):
            self.log('eta = %d.' % eta)
            self.log('# non-empty views with sz < %d = %d' % (eta, np.sum((n_C > 0)*(n_C < eta))))
            self.log('#nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            
            # Compute cost_k and d_k (dest_k) for all k
            ###########################################
            # Proc for computing the cost and dest
            def target_proc(p_num, chunk_sz, n_, S):
                existing_shm_cost = shared_memory.SharedMemory(name=shm_cost_name)
                cost_ = np.ndarray(cost_shape, dtype=cost_dtype, buffer=existing_shm_cost.buf)
                existing_shm_dest = shared_memory.SharedMemory(name=shm_dest_name)
                dest_ = np.ndarray(dest_shape, dtype=dest_dtype, buffer=existing_shm_dest.buf)
                
                start_ind = p_num*chunk_sz
                if p_num == (n_proc-1):
                    end_ind = n_
                else:
                    end_ind = (p_num+1)*chunk_sz
                
                for k in range(start_ind, end_ind):
                    if S is None:
                        k1 = k
                    else:
                        k1 = S[k]
                    cost_[k1], dest_[k1] = cost_of_moving(k1, d_e, neigh_ind[k1], U_[k1], local_param,
                                                          c, n_C, Utilde, eta, eta_max)
                
                existing_shm_cost.close()
                existing_shm_dest.close()
            
            ###########################################
            # Parallel cost and dest computation
            proc = []
            chunk_sz = int(n/n_proc)
            for p_num in range(n_proc):
                proc.append(mp.Process(target=target_proc,
                                       args=(p_num,chunk_sz,n,None),
                                       daemon=True))
                proc[-1].start()
                
            for p_num in range(n_proc):
                proc[p_num].join()
            ###########################################
            
            # Sequential version of above
            # for k in range(n):
            #     cost[k], dest[k] = cost_of_moving(k, d_e, neigh_ind[k], U_[k], local_param,
            #                                       c, n_C, Utilde, eta, eta_max)
            
            # Compute point with minimum cost
            # Compute k and cost^* 
            k = np.argmin(np_cost)
            cost_star = np_cost[k]
            
            self.log('Costs computed when eta = %d.' % eta, log_time=True)
            
            # Loop until minimum cost is inf
            while cost_star < np.inf:
                # Move x_k from cluster s to
                # dest_k and update variables
                s = c[k]
                dest_k = np_dest[k]
                c[k] = dest_k
                n_C[s] -= 1
                n_C[dest_k] += 1
                Clstr[s].remove(k)
                Clstr[dest_k].add(k)
                Utilde[dest_k] = U_[k].union(Utilde[dest_k])
                Utilde[s] = set(itertools.chain.from_iterable(neigh_ind[list(Clstr[s])]))
                
                # Compute the set of points S for which 
                # cost of moving needs to be recomputed
                S = np.where((c==dest_k) | (np_dest==dest_k) | np.any(U[:,list(Clstr[s])].toarray(),1))[0].tolist()
                len_S = len(S)
                
                ###########################################
                # cost, dest update for k in S
                ###########################################
                if len_S > intermed_opts['len_S_thresh']: # do parallel update (almost never true)
                    proc = []
                    n_proc_ = min(n_proc, len_S)
                    chunk_sz = int(len_S/n_proc_)
                    for p_num in range(n_proc_):
                        proc.append(mp.Process(target=target_proc,
                                               args=(p_num,chunk_sz,len_S,S),
                                               daemon=True))
                        proc[-1].start()

                    for p_num in range(n_proc_):
                        proc[p_num].join()
                    ###########################################
                else: # sequential update
                    for k in S:
                        np_cost[k], np_dest[k] = cost_of_moving(k, d_e, neigh_ind[k], U_[k], local_param,
                                                                  c, n_C, Utilde, eta, eta_max)
                ###########################################
                ###########################################
                # Recompute point with minimum cost
                # Recompute k and cost^*
                k = np.argmin(np_cost)
                cost_star = np_cost[k]
            print('Remaining #nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            self.log('Done with eta = %d.' % eta, log_time=True)
        
        self.log('Pruning and cleaning up.')
        del U_
        del Clstr
        del Utilde
        del cost
        del np_cost
        shm_cost.close()
        shm_cost.unlink()
        del dest
        del np_dest
        shm_dest.close()
        shm_dest.unlink()
        
        # Prune empty clusters
        non_empty_C = n_C > 0
        M = np.sum(non_empty_C)
        old_to_new_map = np.arange(n)
        old_to_new_map[non_empty_C] = np.arange(M)
        c = old_to_new_map[c]
        n_C = n_C[non_empty_C]
        
        # Construct a boolean array C s.t. C[m,i] = 1 if c_i == m, 0 otherwise
        C = csr_matrix((np.ones(n), (c, np.arange(n))),
                       shape=(M,n), dtype=bool)
        
        # Compute intermediate views
        intermed_param = copy.deepcopy(local_param)
        if intermed_opts['algo'] == 'LDLE':
            intermed_param.Psi_i = local_param.Psi_i[non_empty_C,:]
            intermed_param.Psi_gamma = local_param.Psi_gamma[non_empty_C,:]
            intermed_param.b = intermed_param.b[non_empty_C]
        elif intermed_opts['algo'] == 'LTSA':
            intermed_param.Psi = local_param.Psi[non_empty_C,:]
            intermed_param.mu = local_param.mu[non_empty_C,:]
            intermed_param.b = intermed_param.b[non_empty_C]
        
        # Compute Utilde_m
        Utilde = C.dot(U)
        
        intermed_param.zeta = np.ones(M);
        for m in range(M):
            Utilde_m = Utilde[m,:].indices
            d_e_Utilde_m = d_e[np.ix_(Utilde_m,Utilde_m)]
            intermed_param.zeta[m] = compute_zeta(d_e_Utilde_m,
                                                  intermed_param.eval_({'view_index': m,
                                                                        'data_mask': Utilde_m}))

        self.log('Done.', log_time=True)
        print("After clustering, max distortion is %f" % (np.max(intermed_param.zeta)))
        self.C = C
        self.c = c
        self.n_C = n_C
        self.Utilde = Utilde
        self.intermed_param = intermed_param