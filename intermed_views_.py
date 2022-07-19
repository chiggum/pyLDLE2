import pdb
import time
import numpy as np
import copy

from util_ import print_log, compute_zeta
from util_ import Param

from scipy.spatial.distance import pdist, squareform

# Computes cost_k, d_k (dest_k)
def cost_of_moving(k, d_e, neigh_ind_k, U_k, local_param, c, n_C,
                   Utilde, eta_min, eta_max, b=None):
    c_k = c[k]
    # Compute |C_{c_k}|
    n_C_c_k = n_C[c_k]
    
    # Check if |C_{c_k}| < eta_{min}
    # If not then c_k is already
    # an intermediate cluster
    if n_C_c_k >= eta_min:
        return np.inf, -1
    
    # Initializations
    n = n_C.shape[0]    
    cost_x_k_to = np.zeros(n) + np.inf
    
    # Compute neighboring clusters c_{U_k} of x_k
    c_U_k = np.unique(c[neigh_ind_k]).tolist()
    
    # Iterate over all m in c_{U_k}
    for m in c_U_k:
        if m == c_k:
            continue
            
        # Compute |C_{m}|
        n_C_m = n_C[m]
        # Check if |C_{m}| < eta_{max}. If not
        # then mth cluster has reached the max
        # allowed size of the cluster. Move on.
        if n_C_m >= eta_max:
            continue
        
        # Check if |C_{m}| >= |C_{c_k}|. If yes, then
        # mth cluster satisfies all required conditions
        # and is a candidate cluster to move x_k in.
        if n_C_m >= n_C_c_k:
            # Compute union of Utilde_m U_k
            U_k_U_Utilde_m = list(U_k.union(Utilde[m]))
            # Compute the cost of moving x_k to mth cluster,
            # that is cost_{x_k \rightarrow m}
            cost_x_k_to[m] = compute_zeta(d_e[np.ix_(U_k_U_Utilde_m,U_k_U_Utilde_m)].toarray(),
                                  local_param.eval_({'view_index': m,
                                                     'data_mask': U_k_U_Utilde_m}))
        
    
    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
        
    return cost_k, dest_k

class IntermedViews:
    def __init__(self, exit_at, print_logs=True, debug=False):
        self.exit_at = exit_at
        self.print_logs = print_logs
        self.debug = debug
        
        self.c = None
        self.C = None
        self.n_C = None
        self.Utilde = None
        self.intermed_param = None
        
        self.local_start_time = time.time()
        self.global_start_time = time.time()
    
    def log(self, s='', log_time=False):
        if self.print_logs:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    def fit(self, d, d_e, neigh_ind, U, local_param, intermed_opts):
        n = d_e.shape[0]
        c = np.arange(n)
        n_C = np.zeros(n) + 1
        Clstr = list(map(set, np.arange(n).reshape((n,1)).tolist()))
        Utilde = list(map(set, neigh_ind.tolist()))
        U_ = list(map(set, neigh_ind.tolist()))
        
        eta_max = intermed_opts['eta_max']
        
        # Vary eta from 2 to eta_{min}
        self.log('Constructing intermediate views.')
        for eta in range(2,intermed_opts['eta_min']+1):
            self.log('eta = %d.' % eta)
            print('# non-empty views with sz < %d = %d' % (eta, np.sum((n_C > 0)*(n_C < eta))))
            print('#nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            
            # Compute cost_k and d_k (dest_k) for all k
            cost = np.zeros(n)+np.inf
            dest = np.zeros(n,dtype='int')-1
            for k in range(n):
                cost[k], dest[k] = cost_of_moving(k, d_e, neigh_ind[k,:], U_[k], local_param,
                                                  c, n_C, Utilde, eta, eta_max)
            # Compute point with minimum cost
            # Compute k and cost^* 
            k = np.argmin(cost)
            cost_star = cost[k]
            
            self.log('Costs computed when eta = %d.' % eta, log_time=True)
            
            # Loop until minimum cost is inf
            while cost_star < np.inf:
                # Move x_k from cluster s to
                # dest_k and update variables
                s = c[k]
                dest_k = dest[k]
                c[k] = dest_k
                n_C[s] -= 1
                n_C[dest_k] += 1
                Clstr[s].remove(k)
                Clstr[dest_k].add(k)
                Utilde[dest_k] = U_[k].union(Utilde[dest_k])
                Utilde[s] = set(neigh_ind[list(Clstr[s]),:].flatten())
                
                # Compute the set of points S for which 
                # cost of moving needs to be recomputed
                S = np.where((c==dest_k) | (dest==dest_k) | np.any(U[:,list(Clstr[s])].toarray(),1))[0].tolist()
                
                # Update cost_k and d_k (dest_k) for k in S
                for k in S:
                    cost[k], dest[k] = cost_of_moving(k, d_e, neigh_ind[k,:], U_[k], local_param,
                                                      c, n_C, Utilde, eta, eta_max)
                
                # Recompute point with minimum cost
                # Recompute k and cost^*
                k = np.argmin(cost)
                cost_star = cost[k]
            print('Remaining #nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            self.log('Done with eta = %d.' % eta, log_time=True)
        
        self.log('Pruning and cleaning up.')
        #del U_
        #del Clstr
        #del Utilde
        
        # Prune empty clusters
        non_empty_C = n_C > 0
        M = np.sum(non_empty_C)
        old_to_new_map = np.arange(n)
        old_to_new_map[non_empty_C] = np.arange(M)
        c = old_to_new_map[c]
        n_C = n_C[non_empty_C]
        
        # Construct a boolean array C s.t. C[m,i] = 1 if c_i == m, 0 otherwise
        C = np.zeros((M,n), dtype=bool)
        C[c, np.arange(n)] = True
        
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
        Utilde = np.zeros((M,n),dtype=bool)
        for m in range(M):
            Utilde[m,:] = U[C[m,:],:].sum(0)
        
        intermed_param.zeta = np.ones(M);
        for m in range(M):
            Utilde_m = Utilde[m,:]
            d_e_Utilde_m = d_e[np.ix_(Utilde_m,Utilde_m)].toarray()
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