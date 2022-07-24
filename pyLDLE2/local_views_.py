import pdb
import time
import numpy as np
import copy

from . import gl_
from . import ipge_
from .util_ import print_log, compute_zeta, to_dense
from .util_ import Param, sparse_matrix

from scipy.linalg import inv, svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

import multiprocess as mp

class LocalViews:
    def __init__(self, exit_at=None, verbose=True, debug=False):
        self.exit_at = exit_at
        self.logs = verbose
        self.debug = debug
        
        
        self.epsilon = None
        self.U = None
        self.local_param_pre = None
        self.local_param_post = None
        
        # For LDLE
        self.GL = None
        self.IPGE = None
        self.gamma = None
        self.phi = None
        
        self.local_start_time = time.time()
        self.global_start_time = time.time()
        
    def log(self, s='', log_time=False):
        if self.logs:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
            
    def fit(self, d, X, d_e, neigh_dist, neigh_ind, ddX, local_opts):
        if local_opts['algo'] == 'LDLE':
            self.log('Constructing ' + local_opts['gl_type'] + ' graph Laplacian + its eigendecomposition.')
            GL = gl_.GL(debug=self.debug)
            GL.fit(neigh_dist, neigh_ind, local_opts)
            self.log('Done.', log_time=True)
            #############################################
            
            # Local views in the ambient space
            epsilon = neigh_dist[:,[local_opts['k']-1]]
            U = sparse_matrix(neigh_ind[:,:local_opts['k']],
                              np.ones((neigh_ind.shape[0],local_opts['k']), dtype=bool))
            #U = U.maximum(U.transpose()) # Idk why I wrote this. This is not needed.
            #############################################

            self.log('Computing Atilde: Inner Prod of Grad of EigFuncs.')
            IPGE = ipge_.IPGE(debug=self.debug)
            if local_opts['Atilde_method'] == 'LLR':
                print('Using LLR')
            elif local_opts['Atilde_method'] == 'LDLE_2':
                print('Using LDLE_2')
            elif local_opts['Atilde_method'] == 'LDLE_3':
                print('Using LDLE_3')
            else: #fem
                IPGE.fem(opts = {'phi': GL.phi, 'neigh_ind': neigh_ind, 'k': local_opts['k'],
                                 'neigh_dist': neigh_dist, 'epsilon': epsilon,
                                 'd': d, 'p': local_opts['p']})
            self.log('Done.', log_time=True)
            
            if self.exit_at == 'post-Atilde':
                return
            #############################################
            
            # Compute gamma
            if local_opts['no_gamma']:
                gamma = np.ones((GL.phi.shape[0], local_opts['N']))
            else:
                gamma = np.sqrt(local_opts['k']/U.dot(GL.phi**2))
            #############################################

            # Compute LDLE: Low Distortion Local Eigenmaps
            self.log('Computing LDLE.')
            local_param_pre = self.compute_LDLE(d, d_e, neigh_ind, neigh_dist, GL.phi, U, IPGE.Atilde, gamma, local_opts)
            self.log('Done.', log_time=True)
            #############################################

            if local_opts['to_postprocess']:
                self.log('Posprocessing LDLE.')
                local_param_post = self.postprocess_LDLE(d_e, neigh_ind, neigh_dist, local_param_pre, U, local_opts)
                self.log('Done.', log_time=True)
            else:
                local_param_post = local_param_pre
            #############################################
            # set b
            local_param_post.b = np.ones(neigh_ind.shape[0])
            for k in range(neigh_ind.shape[0]):
                d_e_k = to_dense(d_e[np.ix_(neigh_ind[k,:],neigh_ind[k,:])])
                Psi_k = local_param_post.eval_({'view_index': k, 'data_mask': neigh_ind[k,:]})
                local_param_post.b[k] = np.median(squareform(d_e_k))/np.median(pdist(Psi_k))
            
            #############################################
            
            if ddX is not None:
                self.log('Halving objects.')
                n = ddX.shape[0]
                U = U[:n,:n]
                GL.phi = GL.phi[:n,:]
                IPGE.Atilde = IPGE.Atilde[:n,:,:]
                gamma = gamma[:n,:]
                
                local_param_pre.Psi_i = local_param_pre.Psi_i[:n,:]
                local_param_pre.Psi_gamma = local_param_pre.Psi_gamma[:n,:]
                local_param_pre.zeta = local_param_pre.zeta[:n]
                
                local_param_post.Psi_i = local_param_post.Psi_i[:n,:]
                local_param_post.Psi_gamma = local_param_post.Psi_gamma[:n,:]
                local_param_post.zeta = local_param_post.zeta[:n]
                local_param_post.b = local_param_post.b[:n]
                
                d_e_ = d_e[:n,:n]
                
                for k in range(n):
                    U_k = U[k,:].toarray().flatten()
                    local_param_pre.zeta[k] = compute_zeta(d_e_[np.ix_(U_k,U_k)],
                                                           local_param_pre.eval_({'view_index': k,
                                                                                  'data_mask': U_k}))
                    local_param_post.zeta[k] = compute_zeta(d_e_[np.ix_(U_k,U_k)],
                                                            local_param_post.eval_({'view_index': k,
                                                                                    'data_mask': U_k}))
                self.log('Done.', log_time=True)
            #############################################
        else:
            self.log('Constructing local views using LTSA.')
            # Local views in the ambient space
            epsilon = neigh_dist[:,[local_opts['k']-1]]
            U = sparse_matrix(neigh_ind[:,:local_opts['k']-1],
                              np.ones((neigh_ind.shape[0],local_opts['k']-1), dtype=bool))
            local_param_pre = None
            local_param_post = self.compute_LTSAP(d, X, d_e, neigh_ind)
            local_param_post.b = np.ones(X.shape[0])
            self.log('Done.', log_time=True)
            
        print('Max local distortion =', np.max(local_param_post.zeta))
        if self.debug:
            if local_opts['algo'] == 'LDLE':
                self.GL = GL
                self.IPGE = IPGE
                self.gamma = gamma
            
            self.epsilon = epsilon
            self.local_param_pre = local_param_pre
        
        if local_opts['algo'] == 'LDLE':
            self.phi = GL.phi
            
        self.U = U
        self.local_param_post = local_param_post
    
    def compute_LDLE(self, d, d_e, neigh_ind, neigh_dist, phi, U, Atilde, gamma, local_opts, print_prop = 0.25):
        n, N = phi.shape
        N = phi.shape[1]
        tau = local_opts['tau']
        delta = local_opts['delta']
        
        print_freq = np.int(n*print_prop)
        
        local_param = Param('LDLE')
        local_param.phi = phi
        local_param.Psi_gamma = np.zeros((n,d))
        local_param.Psi_i = np.zeros((n,d),dtype='int')
        local_param.zeta = np.zeros(n)
        n_proc = local_opts['n_proc']
        
        def target_proc(p_num, chunk_sz, q_):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = n
            else:
                end_ind = (p_num+1)*chunk_sz

            for k in range(start_ind, end_ind):
                # to store i_1, ..., i_d
                i = np.zeros(d, dtype='int')

                # Grab the precomputed U_k, Atilde_{kij}, gamma_{ki}
                U_k = U[k,:]
                Atilde_k = Atilde[k,:,:]
                gamma_k = gamma[k,:]

                # Compute theta_1
                Atikde_kii = Atilde_k.diagonal()
                theta_1 = np.percentile(Atikde_kii, tau)

                # Compute Stilde_k
                Stilde_k = Atikde_kii >= theta_1

                # Compute i_1
                r_1 = np.argmax(Stilde_k) # argmax finds first index with max value
                temp = gamma_k * np.abs(Atilde_k[:,r_1])
                alpha_1 = np.max(temp * Stilde_k)
                i[0] = np.argmax((temp >= delta*alpha_1) & (Stilde_k))

                for s in range(1,d):
                    i_prev = i[0:s]
                    # compute temp variable to help compute Hs_{kij} below
                    temp = inv(Atilde_k[np.ix_(i_prev,i_prev)])

                    # Compute theta_s
                    Hs_kii = Atikde_kii - np.sum(Atilde_k[:,i_prev] * np.dot(temp, Atilde_k[i_prev,:]).T, 1)
                    temp_ = Hs_kii[Stilde_k]
                    theta_s = np.percentile(temp_, tau)

                    #theta_s=np.max([theta_s,np.min([np.max(temp_),1e-4])])

                    # Compute i_s
                    r_s = np.argmax((Hs_kii>=theta_s) & Stilde_k)
                    Hs_kir_s = Atilde_k[:,[r_s]] - np.dot(Atilde_k[:,i_prev],
                                                          np.dot(temp, Atilde_k[i_prev,r_s][:,np.newaxis]))
                    temp = gamma_k * np.abs(Hs_kir_s.flatten())
                    alpha_s = np.max(temp * Stilde_k)
                    i[s]=np.argmax((temp >= delta*alpha_s) & Stilde_k);

                # Compute Psi_k
                local_param.Psi_gamma[k,:] = gamma_k[i]
                local_param.Psi_i[k,:] = i

                # Compute zeta_{kk}
                d_e_k = d_e[np.ix_(neigh_ind[k,:],neigh_ind[k,:])]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': neigh_ind[k,:]}))

            q_.put((start_ind, end_ind,
                    local_param.zeta[start_ind:end_ind],
                    local_param.Psi_gamma[start_ind:end_ind,:],
                    local_param.Psi_i[start_ind:end_ind,:]))
        
        q_ = mp.Queue()
        chunk_sz = int(n/n_proc)
        proc = []
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, zeta_, Psi_gamma_, Psi_i_ = q_.get()
            local_param.zeta[start_ind:end_ind] = zeta_
            local_param.Psi_gamma[start_ind:end_ind,:] = Psi_gamma_
            local_param.Psi_i[start_ind:end_ind,:] = Psi_i_

        q_.close()
        
        for p_num in range(n_proc):
            proc[p_num].join()
            
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        return local_param
    
    def compute_LTSAP(self, d, X, d_e, neigh_ind, print_prop = 0.25):
        n = neigh_ind.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        
        local_param = Param('LTSA')
        local_param.X = X
        local_param.Psi = np.zeros((n,p,d))
        local_param.mu = np.zeros((n,p))
        local_param.zeta = np.zeros(n)

        # iterate over points in the data
        for k in range(n):
            if print_freq and np.mod(k, print_freq)==0:
                print('local_param: %d points processed...' % k)
            
            neigh_ind_k = neigh_ind[k,:]
            # LTSA
            X_k = X[neigh_ind_k,:]
            xbar_k = np.mean(X_k,axis=0)[np.newaxis,:]
            X_k = X_k - xbar_k
            X_k = X_k.T
            if p == d:
                Q_k,Sigma_k,_ = svd(X_k)
            else:
                Q_k,Sigma_k,_ = svds(X_k, d, which='LM')
                
            local_param.Psi[k,:,:] = Q_k[:,:d]
            local_param.mu[k,:] = xbar_k
            
            # Compute zeta_{kk}
            d_e_k = d_e[np.ix_(neigh_ind_k, neigh_ind_k)]
            local_param.zeta[k] = compute_zeta(d_e_k,
                                               local_param.eval_({'view_index': k,
                                                                  'data_mask': neigh_ind_k}))
            
        print('local_param: all %d points processed...' % n)
        return local_param
    
    def postprocess_LDLE(self, d_e, neigh_ind, neigh_dist, local_param_pre, U, local_opts):
        # initializations
        n = neigh_ind.shape[0]
        local_param = copy.deepcopy(local_param_pre)
        
        N_replaced = n
        itr = 1
        # Extra variable to speed up
        param_changed_old = None
        n_proc = local_opts['n_proc']
        
        while N_replaced:
            new_param_of = np.arange(n, dtype='int')
            param_changed_new = np.zeros(n, dtype=bool)
            if N_replaced > local_opts['pp_n_thresh']: # use multiple processors
                zeta = local_param.zeta
                if param_changed_old is not None:
                    param_changed_old = set(np.where(param_changed_old)[0])
                    
                def target_proc(p_num, chunk_sz, q_):
                    start_ind = p_num*chunk_sz
                    if p_num == (n_proc-1):
                        end_ind = n
                    else:
                        end_ind = (p_num+1)*chunk_sz

                    for k in range(start_ind, end_ind):
                        U_k = U[k,:].nonzero()[1].tolist()
                        if param_changed_old is None:
                            cand_k = U_k
                        else:
                            cand_k = param_changed_old.intersection(U_k)
                        neigh_ind_k = neigh_ind[k,:]
                        d_e_k = d_e[np.ix_(neigh_ind_k,neigh_ind_k)]
                        for kp in cand_k:
                            Psi_kp_on_U_k = local_param.eval_({'view_index': kp,
                                                               'data_mask': neigh_ind_k})
                            zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)

                            # if zeta_{kk'} < zeta_{kk}
                            if zeta_kkp < zeta[k]:
                                zeta[k] = zeta_kkp
                                new_param_of[k] = kp
                                param_changed_new[k] = 1

                    q_.put((start_ind, end_ind, zeta[start_ind:end_ind],
                            new_param_of[start_ind:end_ind],
                            param_changed_new[start_ind:end_ind]))
                
                q_ = mp.Queue()
                chunk_sz = int(n/n_proc)
                proc = []
                for p_num in range(n_proc):
                    proc.append(mp.Process(target=target_proc,
                                           args=(p_num,chunk_sz,q_),
                                           daemon=True))
                    proc[-1].start()

                for p_num in range(n_proc):
                    start_ind, end_ind, zeta_, new_param_of_, param_changed_new_ = q_.get()
                    zeta[start_ind:end_ind] = zeta_
                    param_changed_new[start_ind:end_ind] = param_changed_new_
                    new_param_of[start_ind:end_ind] = new_param_of_
                
                q_.close()
                
                for p_num in range(n_proc):
                    proc[p_num].join()
                    
                local_param.zeta = zeta
            else: # do sequentially (single processor)
                # Iterate over all local parameterizations
                # To speed up the process, only consider those neighbors
                # for which the parameterization changed in the prev step
                if param_changed_old is None:
                    cand = U.tocoo()
                else:
                    cand = U.multiply(param_changed_old[None,:])
                    cand.eliminate_zeros()

                cand_i = cand.row
                cand_j = cand.col
                n_ind = cand_i.shape[0]
                ind = 0
                for k in range(n):
                    cand_k = []
                    while (ind < n_ind) and (k == cand_i[ind]):
                        cand_k.append(cand_j[ind])
                        ind += 1

                    neigh_ind_k = neigh_ind[k,:]
                    d_e_k = d_e[np.ix_(neigh_ind_k,neigh_ind_k)]
                    for kp in cand_k:
                        Psi_kp_on_U_k = local_param.eval_({'view_index': kp,
                                                           'data_mask': neigh_ind_k})
                        zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)

                        # if zeta_{kk'} < zeta_{kk}
                        if zeta_kkp < local_param.zeta[k]:
                            local_param.zeta[k] = zeta_kkp
                            new_param_of[k] = kp
                            param_changed_new[k] = 1
            
            local_param.Psi_i = local_param.Psi_i[new_param_of,:]
            local_param.Psi_gamma = local_param.Psi_gamma[new_param_of,:]
            param_changed_old = param_changed_new.copy()
            N_replaced = np.sum(param_changed_new)
            
            print("Iter %d, Param replaced: %d, max distortion: %f" % (itr, N_replaced, np.max(local_param.zeta)))
            itr = itr + 1
        
        return local_param