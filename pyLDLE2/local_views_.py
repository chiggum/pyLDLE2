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
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.sparse import coo_matrix, csr_matrix

import multiprocess as mp
from multiprocess import shared_memory

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
            if local_opts['scale_by']=='none':
                gamma = np.ones((GL.phi.shape[0], local_opts['N']))
            elif local_opts['scale_by']=='gamma':
                gamma = 1/(np.sqrt(U.dot(GL.phi**2)/local_opts['k'])+1e-12)
            else:
                gamma = np.repeat(np.power(GL.lmbda.flatten(),local_opts['scale_by']),
                                  GL.phi.shape[0]).reshape(GL.phi.shape)
            #############################################

            # Compute LDLE: Low Distortion Local Eigenmaps
            self.log('Computing LDLE.')
            local_param_pre = self.compute_LDLE(d, d_e, neigh_dist, GL.phi, U, IPGE.Atilde, gamma, local_opts)
            self.log('Done.', log_time=True)
            #############################################

            if local_opts['to_postprocess']:
                self.log('Posprocessing local parameterizations.')
                local_param_post = self.postprocess(d_e, neigh_dist, neigh_ind, local_param_pre, U, local_opts)
                self.log('Done.', log_time=True)
            else:
                local_param_post = local_param_pre
            #############################################
            # set b
            local_param_post.b = np.ones(U.shape[0])
            n_proc = local_opts['n_proc']
            n_ = U.shape[0]
            chunk_sz = int(n_/n_proc)
            def target_proc(p_num, q_):
                start_ind = p_num*chunk_sz
                if p_num == (n_proc-1):
                    end_ind = n_
                else:
                    end_ind = (p_num+1)*chunk_sz
                b_ = np.ones(end_ind-start_ind)
                for k in range(start_ind, end_ind):
                    U_k = U[k,:].indices
                    d_e_k = to_dense(d_e[np.ix_(U_k,U_k)])
                    Psi_k = local_param_post.eval_({'view_index': k, 'data_mask': U_k})
                    b_[k-start_ind] = np.median(squareform(d_e_k))/np.median(pdist(Psi_k))
                q_.put((start_ind, end_ind, b_))
            
            q_ = mp.Queue()
            proc = []
            for p_num in range(n_proc):
                proc.append(mp.Process(target=target_proc, args=(p_num, q_)))
                proc[-1].start()
                
            for p_num in range(n_proc):
                start_ind, end_ind, b_ = q_.get()
                local_param_post.b[start_ind:end_ind] = b_
            q_.close()
            for p_num in range(n_proc):
                proc[p_num].join()
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
            self.log('Constructing local views using ' + local_opts['algo'])
            # Local views in the ambient space
            if local_opts['U_method'] == 'k_nn':
                U = sparse_matrix(neigh_ind[:,:local_opts['k']],
                                  np.ones((neigh_ind.shape[0],local_opts['k']), dtype=bool))
            else:
                neigh_ind_ = []
                neigh_dist_ = []
                radius = neigh_dist[:,local_opts['k']-1].max()
                for k in range(neigh_ind.shape[0]):
                    mask = neigh_dist[k] < radius
                    neigh_ind_.append(neigh_ind[k][mask])
                    neigh_dist_.append(np.ones(np.sum(mask), dtype=bool))
                U = sparse_matrix(np.array(neigh_ind_), np.array(neigh_dist_))
                    
            if local_opts['algo'] == 'Smooth-LTSA':
                local_param_pre = self.compute_Smooth_LTSAP(d, X, d_e, U, local_opts)
            else:
                local_param_pre = self.compute_LTSAP(d, X, d_e, U, local_opts)
            self.log('Done.', log_time=True)
            if local_opts['to_postprocess']:
                self.log('Posprocessing local parameterizations.')
                local_param_post = self.postprocess(d_e, neigh_dist, neigh_ind, local_param_pre, U, local_opts)
                self.log('Done.', log_time=True)
            else:
                local_param_post = local_param_pre
            local_param_post.b = np.ones(X.shape[0])
            
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
    
    def compute_LDLE(self, d, d_e, neigh_dist, phi, U, Atilde, gamma, local_opts, print_prop = 0.25):
        n, N = phi.shape
        N = phi.shape[1]
        tau = local_opts['tau']
        delta = local_opts['delta']
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
                U_k = U[k,:].indices
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
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))

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
    
    def compute_LTSAP(self, d, X, d_e, U, local_opts, print_prop = 0.25):
        n = U.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        
        local_param = Param('LTSA')
        local_param.X = X
        local_param.Psi = np.zeros((n,p,d))
        local_param.mu = np.zeros((n,p))
        local_param.zeta = np.zeros(n)
        n_proc = local_opts['n_proc']
        
        def target_proc(p_num, chunk_sz, q_):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = n
            else:
                end_ind = (p_num+1)*chunk_sz

            for k in range(start_ind, end_ind):
                U_k = U[k,:].indices
                # LTSA
                X_k = X[U_k,:]
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
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))
            
            q_.put((start_ind, end_ind,
                    local_param.zeta[start_ind:end_ind],
                    local_param.Psi[start_ind:end_ind,:],
                    local_param.mu[start_ind:end_ind,:]))
        
        q_ = mp.Queue()
        chunk_sz = int(n/n_proc)
        proc = []
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, zeta_, Psi_, mu_ = q_.get()
            local_param.zeta[start_ind:end_ind] = zeta_
            local_param.Psi[start_ind:end_ind,:] = Psi_
            local_param.mu[start_ind:end_ind,:] = mu_

        q_.close()
        
        for p_num in range(n_proc):
            proc[p_num].join()
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        return local_param
    
    def compute_Smooth_LTSAP(self, d, X, d_e, U, local_opts, print_prop = 0.25):
        local_param_ = self.compute_LTSAP(d, X, d_e, U, local_opts)
        zeta0 = local_param_.zeta.copy()
        del local_param_
        
        n = U.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        max_iter = local_opts['max_iter']
        alpha = local_opts['alpha']
        reg = local_opts['reg']
        
        def grad_1(Q, Sigma, Q_tilde):
            return -2*Sigma.dot(Q) 
        
        def grad_2(Q, Sigma, Q_tilde):
            if reg == 0:
                return 0
            return -reg*2*Q_tilde
        
        def grad_(Q, Sigma, Q_tilde):
            return grad_1(Q, Sigma, Q_tilde) + grad_2(Q, Sigma, Q_tilde)
        
        def skew(A):
            return 0.5*(A-A.T)
        
        def proj_(A, Q):
            return (np.eye(p)-Q.dot(Q.T)).dot(A) + Q.dot(skew(Q.T.dot(A)))
        
        def obj_val_1(Q, Sigma, Q_tilde):
            return np.trace((np.eye(p)-Q.dot(Q.T)).dot(Sigma))
        
        def obj_val_2(Q, Sigma, Q_tilde):
            if reg == 0:
                return 0
            return reg*(2*d - 2*np.trace(Q.dot(Q_tilde.T)))
        
        def obj_val(Q, Sigma, Q_tilde):
            return obj_val_1(Q, Sigma, Q_tilde)+obj_val_2(Q, Sigma, Q_tilde)
        
        def unique_qr(A):
                Q, R = np.linalg.qr(A)
                signs = 2 * (np.diag(R) >= 0) - 1
                Q = Q * signs[np.newaxis, :]
                R = R * signs[:, np.newaxis]
                return Q, R
        
        local_param = Param('LTSA')
        local_param.X = X
        local_param.Psi = np.zeros((n,p,d))
        local_param.mu = np.zeros((n,p))
        local_param.zeta = np.zeros(n)+np.inf
        
        n_U_U = U.astype(int).dot(U.astype(int).transpose())
        # Compute maximum spanning tree/forest of W
        T = minimum_spanning_tree(-n_U_U)
        # Detect clusters of manifolds and create
        # a sequence of intermediate views for each of them
        n_visited = 0
        seq_of_local_views_in_cluster = []
        parents_of_local_views_in_cluster = []
        # stores cluster number for the intermediate views in a cluster
        cluster_of_local_view = np.zeros(n,dtype=int)
        is_visited = np.zeros(n, dtype=int)
        cluster_num = 0
        while n_visited < n:
            # First intermediate view in the sequence
            s_1 = np.argmin(zeta0 + 100000000*is_visited)
            # Compute breadth first order in T starting from s_1
            s_, rho_ = breadth_first_order(T, s_1, directed=False) #(ignores edge weights)
            seq_of_local_views_in_cluster.append(s_)
            parents_of_local_views_in_cluster.append(rho_)
            is_visited[s_] = True
            cluster_of_local_view[s_] = cluster_num
            n_visited = np.sum(is_visited)
            cluster_num = cluster_num + 1
        
        ctr = 0
        for i in range(cluster_num):
            seq = seq_of_local_views_in_cluster[i]
            rho = parents_of_local_views_in_cluster[i]
            for ki in range(seq.shape[0]):
                k = seq[ki]
                U_k = U[k,:].indices
                X_k = X[U_k,:]
                xbar_k = np.mean(X_k,axis=0)[np.newaxis,:]
                X_k = X_k - xbar_k
                X_k = X_k.T
                n_k = X_k.shape[1]
                if ki == 0:
                    if p == d:
                        Q_k,Sigma_k,_ = svd(X_k)
                    else:
                        Q_k,Sigma_k,_ = svds(X_k, d, which='LM')
                    Q_k = Q_k[:,:d]
                else:
                    Q_tilde = local_param.Psi[rho[k],:,:]
                    Q_k = Q_tilde.copy()
                    Sigma = X_k.dot(X_k.transpose())
                    #pdb.set_trace()
                    if ctr%print_freq == 1:
                        print('Starting objective val:',
                              obj_val_1(Q_k, Sigma, Q_tilde),
                              obj_val_2(Q_k, Sigma, Q_tilde))
                        print('Starting Q_k[0,:]:', Q_k[0,:])
                        print('Starting proj(grad)[0,:]:', proj_(grad_(Q_k, Sigma, Q_tilde), Q_k)[0,:])
                        
                    for _ in range(max_iter):
                        step_ = proj_(grad_(Q_k, Sigma, Q_tilde), Q_k)
                        if np.mean(np.abs(step_)) <  1e-6:
                            break
                        Q_k,R_k = unique_qr(Q_k - alpha*step_)
                    if ctr%print_freq == 1:
                        print('Ending objective val:',
                              obj_val_1(Q_k, Sigma, Q_tilde),
                              obj_val_2(Q_k, Sigma, Q_tilde))
                        print('Ending Q_k[0,:]:', Q_k[0,:])
                        print('Ending proj(grad)[0,:]:', proj_(grad_(Q_k, Sigma, Q_tilde), Q_k)[0,:])
                    #pdb.set_trace()
                local_param.Psi[k,:,:] = Q_k.copy()
                local_param.mu[k,:] = xbar_k
                # Compute zeta_{kk}
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))
                if ctr%print_freq == 1:
                    print('local_param: %d points processed' % ctr)
                    print('#'*50)
                ctr += 1
                
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        return local_param
    
    def postprocess(self, d_e, neigh_dist, neigh_ind, local_param_pre, U, local_opts):
        # initializations
        n = U.shape[0]
        local_param = copy.deepcopy(local_param_pre)

        n_proc = local_opts['n_proc']
        barrier = mp.Barrier(n_proc)
        pcb = np.zeros(n, dtype=bool) # param changed buffer and converge flag
        npo = np.arange(n, dtype=int) # new param of
        zeta = local_param.zeta

        pcb_dtype = pcb.dtype
        pcb_shape = pcb.shape
        npo_dtype = npo.dtype
        npo_shape = npo.shape
        zeta_shape = zeta.shape
        zeta_dtype = zeta.dtype

        shm_pcb = shared_memory.SharedMemory(create=True, size=pcb.nbytes)
        np_pcb = np.ndarray(pcb_shape, dtype=pcb_dtype, buffer=shm_pcb.buf)
        np_pcb[:] = pcb[:]
        shm_npo = shared_memory.SharedMemory(create=True, size=npo.nbytes)
        np_npo = np.ndarray(npo_shape, dtype=npo_dtype, buffer=shm_npo.buf)
        np_npo[:] = npo[:]
        shm_zeta = shared_memory.SharedMemory(create=True, size=zeta.nbytes)
        np_zeta = np.ndarray(zeta_shape, dtype=zeta_dtype, buffer=shm_zeta.buf)
        np_zeta[:] = zeta[:]

        shm_pcb_name = shm_pcb.name
        shm_npo_name = shm_npo.name
        shm_zeta_name = shm_zeta.name

        def target_proc(p_num, chunk_sz, barrier, U, neigh_ind, local_param, d_e):
            existing_shm_pcb = shared_memory.SharedMemory(name=shm_pcb_name)
            param_changed_buf = np.ndarray(pcb_shape, dtype=pcb_dtype,
                                           buffer=existing_shm_pcb.buf)
            existing_shm_npo = shared_memory.SharedMemory(name=shm_npo_name)
            new_param_of = np.ndarray(npo_shape, dtype=npo_dtype,
                                      buffer=existing_shm_npo.buf)
            existing_shm_zeta = shared_memory.SharedMemory(name=shm_zeta_name)
            zeta_ = np.ndarray(zeta_shape, dtype=zeta_dtype,
                                      buffer=existing_shm_zeta.buf)

            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = n
            else:
                end_ind = (p_num+1)*chunk_sz

            param_changed_old = None
            new_param_of_ = np.arange(start_ind, end_ind)
            N_replaced = n
            while N_replaced: # while not converged
                for k in range(start_ind, end_ind):
                    param_changed_for_k = False
                    U_k = U[k,:].indices
                    # TODO: which one of the two should be used?
                    neigh_ind_k = U_k # theoretically sound.
                    # neigh_ind_k = neigh_ind[k,:] # ask for low distortion on slightly bigger views
                    if param_changed_old is None:
                        cand_k = U_k.tolist()
                    else:
                        cand_k = list(param_changed_old.intersection(U_k.tolist()))
                    if len(cand_k)==0:
                        param_changed_buf[k] = False
                        continue
                    d_e_k = d_e[np.ix_(neigh_ind_k,neigh_ind_k)]
                    
                    for kp in cand_k:
                        Psi_kp_on_U_k = local_param.eval_({'view_index': new_param_of[kp],
                                                           'data_mask': neigh_ind_k})
                        zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)
                        # if zeta_{kk'} < zeta_{kk}
                        if zeta_kkp < zeta_[k]:
                            zeta_[k] = zeta_kkp
                            new_param_of_[k-start_ind] = new_param_of[kp]
                            param_changed_for_k = True
                    param_changed_buf[k] = param_changed_for_k
                
                barrier.wait()
                new_param_of[start_ind:end_ind] = new_param_of_
                param_changed_old = set(np.where(param_changed_buf)[0])
                N_replaced = len(param_changed_old)
                barrier.wait()
                if p_num == 0:
                    print("#Param replaced: %d, max distortion: %f" % (N_replaced, np.max(zeta_)))
                        
            existing_shm_pcb.close()
            existing_shm_npo.close()
            existing_shm_zeta.close()

        proc = []
        chunk_sz = int(n/n_proc)
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc, args=(p_num,chunk_sz, barrier,
                                                             U, neigh_ind, local_param, d_e),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            proc[p_num].join()

        npo[:] = np_npo[:]
        local_param.zeta[:] = np_zeta[:]

        del np_npo
        shm_npo.close()
        shm_npo.unlink()
        del np_zeta
        shm_zeta.close()
        shm_zeta.unlink()
        del np_pcb
        shm_pcb.close()
        shm_pcb.unlink()

        if local_opts['algo'] == 'LDLE':
            local_param.Psi_i = local_param.Psi_i[npo,:]
            local_param.Psi_gamma = local_param.Psi_gamma[npo,:]
        else:
            local_param.Psi = local_param.Psi[npo,:]
            local_param.mu = local_param.mu[npo,:]
            
        print('Max local distortion after postprocessing:', np.max(local_param.zeta))
        return local_param