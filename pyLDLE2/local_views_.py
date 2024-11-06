import pdb
import time
import numpy as np
import copy

from . import gl_
from . import ipge_
from .util_ import print_log, compute_zeta, to_dense
from .util_ import Param, sparse_matrix

from .l1pca_optimal_ import l1pca_optimal

from scipy.linalg import inv, svd, pinv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.sparse import coo_matrix, csr_matrix

from sklearn.decomposition import SparsePCA
from hyperspy.learn.rpca import rpca_godec

import multiprocess as mp
from multiprocess import shared_memory

from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA

import itertools

import torch
import torch.nn as nn

def scipy_coo_to_torch_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

# class Model(torch.nn.Module):
#     def __init__(self, cov, b, L, lmbda=1, device='cpu'):
#         super(Model, self).__init__()
#         self.N, self.n, self.p = b.shape
#         Psi = torch.zeros((self.N, self.n, self.p), device=device)
#         self.Psi = torch.nn.Parameter(Psi, requires_grad=True)
#         #self.L = torch.tensor(L.astype('float32')).to(device)
#         self.L = scipy_coo_to_torch_sparse(L).to(device)
#         self.lmbda = lmbda
#         self.cov = []
#         for k in range(self.n):
#             self.cov.append(torch.tensor(cov[k].astype('float32')).to(device))
#         self.b = torch.tensor(b.astype('float32')).to(device)
        
#     def forward(self):
#         loss = 0
#         for j in range(self.p):
#             loss += self.lmbda*torch.sum(torch.sparse.mm(self.L, self.Psi[:,:,j].T)*self.Psi[:,:,j].T)
        
#         loss = loss - 2*torch.sum(self.Psi*self.b)
        
#         for k in range(self.n):
#             loss += torch.sum(torch.matmul(self.Psi[:,k,:], self.cov[k])*self.Psi[:,k,:])
#         return loss

class Model(torch.nn.Module):
    def __init__(self, X_tilde, Phi_tilde, L, Psi=None, lmbda=1, device='cpu'):
        super(Model, self).__init__()
        self.n = len(X_tilde)
        self.p = X_tilde[0].shape[0]
        self.N = Phi_tilde[0].shape[0]
        if Psi is None:
            Psi = torch.zeros((self.N, self.n, self.p), device=device)
        else:
            Psi = torch.tensor(Psi.copy().astype('float32')).to(device)
        self.Psi = torch.nn.Parameter(Psi, requires_grad=True)
        self.L = scipy_coo_to_torch_sparse(L).to(device)
        # self.L_row_inds = L.row
        # self.L_col_inds = L.col
        # self.L_vals = L.data
        self.lmbda = lmbda
        self.X_tilde = []
        self.Phi_tilde = []
        for k in range(self.n):
            self.X_tilde.append(torch.tensor(X_tilde[k].astype('float32')).to(device))
            self.Phi_tilde.append(torch.tensor(Phi_tilde[k].astype('float32')).to(device))
        
    def forward(self):
        loss = 0
        if self.lmbda > 0:
            for j in range(self.p):
                loss += self.lmbda*torch.sum(torch.sparse.mm(self.L, self.Psi[:,:,j].T)*self.Psi[:,:,j].T)
            # for k in range(len(self.L_row_inds)):
            #     i = self.L_row_inds[k]
            #     j = self.L_col_inds[k]
            #     Wij = -self.L_vals[k]
            #     v1 = self.Psi[:,j,:] - self.Psi[:,i,:]
            #     v2 = self.Psi[:,j,:] + self.Psi[:,i,:]
            #     v1_norm = torch.linalg.vector_norm(v1, dim=-1)
            #     v2_norm = torch.linalg.vector_norm(v2, dim=-1)
            #     loss += Wij*torch.sum(torch.minimum(v1_norm, v2_norm))
        
        for k in range(self.n):
            loss += torch.sum((torch.matmul(self.Psi[:,k,:],self.X_tilde[k]) - self.Phi_tilde[k])**2)

        loss = loss/(self.N*self.n)
        return loss

class NNModelBase(torch.nn.Module):
    def __init__(self, p, h_size, n_layers, device='cuda'):
        super(NNModelBase, self).__init__()
        self.layers = []
        self.params = []
        for j in range(n_layers):
            if j == 0:
                self.layers.append(nn.Linear(p, h_size).to(device))
                self.params += list(self.layers[-1].parameters())
                self.layers.append(nn.ReLU().to(device))
            elif j == n_layers-1:
                self.layers.append(nn.Linear(h_size, 1).to(device))
                self.params += list(self.layers[-1].parameters())
            else:
                self.layers.append(nn.Linear(h_size, h_size).to(device))
                self.params += list(self.layers[-1].parameters())
                self.layers.append(nn.ReLU().to(device))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x

class NNModel:
    def __init__(self, X, phi, h_size=128, n_layers=3, batch_size=32,
                 n_epochs=100, print_freq=1, device='cuda', lr=0.01):
        p = X.shape[1]
        n_models = phi.shape[1]
        models = []
        loss_fns = []
        optimizers = []
        for i in range(n_models):
            models.append(NNModelBase(p, h_size, n_layers).to(device))
            loss_fns.append(nn.MSELoss())
            optimizers.append(torch.optim.Adam(models[-1].params, lr=lr))
                
        X_ = torch.tensor(X.astype('float32'), device=device)
        phi_ = torch.tensor(phi.astype('float32'), device=device)

        np.random.seed(42)
        n_batches = X.shape[0]//batch_size + 1
        for i in range(n_epochs):
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            cur_loss = 0
            for j in range(n_batches):
                if j == (n_batches-1):
                    batch = inds[j*batch_size:]
                else:
                    batch = inds[j*batch_size:(j+1)*batch_size]
                for k in range(len(models)):
                    y_pred = models[k](X_[batch,:])
                    loss = loss_fns[k](y_pred, phi_[batch,k:k+1])
                    loss = loss/batch_size
                    optimizers[k].zero_grad()
                    loss.backward()
                    optimizers[k].step()
                    cur_loss += loss.item()
        
            if i%print_freq==0:
                print('epoch', i+1, 'loss:', cur_loss/(n_batches*len(models)))

        Psi = np.zeros((phi.shape[1], X.shape[0], X.shape[1]))
        for i in range(len(models)):
            X_ = torch.tensor(X.astype('float32'), device=device)
            X_.requires_grad = True
            y_pred = models[i](X_).sum()
            y_pred.backward(retain_graph=True)
            Psi[i,:] = X_.grad.detach().cpu().numpy()

        self.Psi = Psi


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
        
        self.local_start_time = time.perf_counter()
        self.global_start_time = time.perf_counter()
        
    def log(self, s='', log_time=False):
        if self.logs:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    # TODO: relax X to be a distance matrix
    def fit(self, d, X, d_e, neigh_dist, neigh_ind, ddX, local_opts):
        print('Computing local views using', local_opts['algo'], flush=True)
        if local_opts['algo'] in ['LDLE', 'LEPC', 'Smooth-LPCA']:
            self.log('Constructing ' + local_opts['gl_type'] + ' graph Laplacian + its eigendecomposition.')
            GL = gl_.GL(debug=self.debug)
            GL.fit(neigh_dist, neigh_ind, local_opts)
            self.log('Done.', log_time=True)
        
        if local_opts['algo'] == 'LDLE' or local_opts['algo'] == 'LEPC':
            #############################################
            
            # Local views in the ambient space
            epsilon = neigh_dist[:,[local_opts['k']-1]]
            U = sparse_matrix(neigh_ind[:,:local_opts['k']],
                              np.ones((neigh_ind.shape[0],local_opts['k']), dtype=bool))
            #U = U.maximum(U.transpose()) # Idk why I wrote this. This is not needed.
            #############################################
            if local_opts['algo'] == 'LDLE':
                self.log('Computing Atilde: Inner Prod of Grad of EigFuncs.')
                IPGE = ipge_.IPGE(debug=self.debug)
                if local_opts['Atilde_method'] == 'LLR':
                    print('Using LLR')
                elif local_opts['Atilde_method'] == 'FeymanKac':
                    print('Using Feyman-Kac formula')
                    IPGE.FeymanKac(GL.L, GL.phi, U)
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
                elif local_opts['scale_by']=='grad_norm':
                    gamma = 1/(np.sqrt(np.diagonal(IPGE.Atilde, axis1=1, axis2=2))+1e-12)
                else:
                    gamma = np.repeat(np.power(1-GL.lmbda.flatten(),local_opts['scale_by']),
                                      GL.phi.shape[0]).reshape(GL.phi.shape)
                #############################################
    
                # Compute LDLE: Low Distortion Local Eigenmaps
                self.log('Computing LDLE.')
                local_param_pre = self.compute_LDLE(d, d_e, GL.phi, U, IPGE.Atilde, gamma, local_opts)
                self.log('Done.', log_time=True)
                #############################################
            elif local_opts['algo'] == 'LEPC':
                local_param_pre = self.compute_LEPC(d, X, d_e, GL.phi, U, GL.L, local_opts)

            if local_opts['to_postprocess']:
                self.log('Posprocessing local parameterizations.')
                local_param_post = self.postprocess(d_e, local_param_pre, U, local_opts)
                self.log('Done.', log_time=True)
            else:
                local_param_post = local_param_pre
            #############################################
            if local_opts['algo'] == 'LDLE':
                local_param_post.gamma = gamma
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
            
            if local_opts['algo'] == 'LDLE' and ddX is not None:
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
        elif local_opts['algo'] == 'RFFLE':
            # Local views in the ambient space
            epsilon = neigh_dist[:,[local_opts['k']-1]]
            U = sparse_matrix(neigh_ind[:,:local_opts['k']],
                              np.ones((neigh_ind.shape[0],local_opts['k']), dtype=bool))
            #U = U.maximum(U.transpose()) # Idk why I wrote this. This is not needed.
            #############################################
            np.random.seed(42)
            rff_v = np.random.normal(0, 1, (local_opts['N'], X.shape[1]))
            rff_xi = np.random.normal(0, 1, (local_opts['N'], 1))
            phi = np.cos(X.dot(rff_v.T) + rff_xi.T)
            #############################################
            self.log('Computing Atilde: Inner Prod of Grad of RFF.')
            IPGE = ipge_.IPGE(debug=self.debug)
            if local_opts['Atilde_method'] == 'LLR':
                print('Using LLR')
            elif local_opts['Atilde_method'] == 'FeymanKac':
                print('Using Feyman-Kac formula')
                self.log('Constructing ' + local_opts['gl_type'] + ' graph Laplacian + its eigendecomposition.')
                GL = gl_.GL(debug=self.debug)
                GL.fit(neigh_dist, neigh_ind, local_opts)
                self.log('Done.', log_time=True)
                IPGE.FeymanKac(GL.L, phi, U)
            elif local_opts['Atilde_method'] == 'LDLE_3':
                print('Using LDLE_3')
            else: #fem
                IPGE.fem(opts = {'phi': phi, 'neigh_ind': neigh_ind, 'k': local_opts['k'],
                                 'neigh_dist': neigh_dist, 'epsilon': epsilon,
                                 'd': d, 'p': local_opts['p']})
            self.log('Done.', log_time=True)
            
            if self.exit_at == 'post-Atilde':
                return
            #############################################
            # Compute gamma
            if local_opts['scale_by']=='none':
                gamma = np.ones((phi.shape[0], local_opts['N']))
            elif local_opts['scale_by']=='gamma':
                gamma = 1/(np.sqrt(U.dot(phi**2)/local_opts['k'])+1e-12)
            elif local_opts['scale_by']=='grad_norm':
                gamma = 1/(np.sqrt(np.diagonal(IPGE.Atilde, axis1=1, axis2=2))+1e-12)
            else:
                gamma = np.ones((phi.shape[0], local_opts['N']))
            #############################################
            # Compute  Random Fourier Features based Local Embedding
            self.log('Computing RFFLE.')
            local_param_pre = self.compute_RFFLE(d, d_e, phi, U, IPGE.Atilde, gamma, local_opts)
            self.log('Done.', log_time=True)
            #############################################

            if local_opts['to_postprocess']:
                self.log('Posprocessing local parameterizations.')
                local_param_post = self.postprocess(d_e, local_param_pre, U, local_opts)
                self.log('Done.', log_time=True)
            else:
                local_param_post = local_param_pre
            #############################################
            local_param_post.gamma = gamma
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
        else:
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
                    
            if local_opts['algo'] == 'Smooth-LPCA':
                local_param_pre = self.compute_Smooth_LPCA(d, X, d_e, U, GL.L, local_opts)
            elif local_opts['algo'] == 'LISOMAP':
                local_param_pre = self.compute_LISOMAP(d, X, d_e, U, local_opts)
            elif local_opts['algo'] == 'LKPCA':
                local_param_pre = self.compute_LKPCA(d, X, d_e, U, local_opts)
            elif local_opts['algo'] == 'EXTPCA':
                local_param_pre = self.compute_EXTPCA(d, X, d_e, U, local_opts)
            else:
                local_param_pre = self.compute_LPCA(d, X, d_e, U, local_opts)
            self.log('Done.', log_time=True)
            if local_opts['to_postprocess']:
                self.log('Posprocessing local parameterizations.')
                local_param_post = self.postprocess(d_e, local_param_pre, U, local_opts)
                self.log('Done.', log_time=True)
            else:
                local_param_post = local_param_pre
            local_param_post.b = np.ones(X.shape[0])
            
        print('Max local distortion =', np.max(local_param_post.zeta))
        if self.debug:
            if local_opts['algo'] == 'LDLE' or local_opts['algo'] == 'RFFLE':
                self.IPGE = IPGE
                self.gamma = gamma
                self.epsilon = epsilon
            self.local_param_pre = local_param_pre
            if local_opts['algo'] == 'LDLE' or local_opts['algo'] == 'LEPC':
                self.GL = GL
            if local_opts['algo'] == 'RFFLE':
                self.rff_v = rff_v
                self.rff_xi = rff_xi
        
        if local_opts['algo'] == 'LDLE':
            self.phi = GL.phi
        elif local_opts['algo'] == 'RFFLE':
            self.phi = phi
            
        self.U = U
        self.local_param_post = local_param_post
    
    def compute_LDLE(self, d, d_e, phi, U, Atilde, gamma, local_opts, print_prop = 0.25):
        n, N = phi.shape
        N = phi.shape[1]
        if local_opts['brute_force'] and (d==2):
            all_pairs = [list(i) for i in itertools.combinations(np.arange(N).tolist(), d)]
        else:
            all_pairs = None
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
                # Grab the precomputed U_k, Atilde_{kij}, gamma_{ki}
                U_k = U[k,:].indices
                gamma_k = gamma[k,:]
                d_e_k = d_e[np.ix_(U_k, U_k)]
                
                if all_pairs is not None:
                    zeta_min = np.inf
                    zeta_min_ind = None
                    
                    for i in all_pairs:
                        # Compute Psi_k
                        local_param.Psi_gamma[k,:] = gamma_k[i]
                        local_param.Psi_i[k,:] = i

                        # Compute zeta_{kk}
                        
                        zeta_ = compute_zeta(d_e_k, local_param.eval_({'view_index': k,
                                                                        'data_mask': U_k}))
                        if zeta_min > zeta_:
                            zeta_min = zeta_
                            zeta_min_ind = i
                     
                    local_param.Psi_gamma[k,:] = gamma_k[zeta_min_ind]
                    local_param.Psi_i[k,:] = zeta_min_ind
                    local_param.zeta[k] = zeta_min
                else:
                    # to store i_1, ..., i_d
                    i = np.zeros(d, dtype='int')
                    
                    Atilde_k = Atilde[k,:,:]

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

    def compute_LEPC(self, d, X, d_e, phi, U, L, local_opts, print_prop = 0.25):
        n, N = phi.shape
        p = X.shape[1]
        print_freq = int(print_prop * n)
        N = phi.shape[1]
        local_param = Param('LPCA')
        local_param.X = X
        local_param.Psi = np.zeros((n,p,d))
        local_param.mu = np.zeros((n,p))
        local_param.zeta = np.zeros(n)
        var_explained = np.zeros((n,p))
        n_pc_dir_chosen = np.zeros(X.shape[0])
        n_proc = local_opts['n_proc']

        # Psi = None
        # local_param_ = self.compute_LPCA(d, X, d_e, U, local_opts)
        # Psi = np.zeros((N, n, p))
        # for k in range(n):
        #     U_k = U[k,:].indices
        #     phi_k = phi[U_k,:] - phi[k,:][None,:]
        #     X_U_k = local_param_.eval_({'view_index': k, 'data_mask': U_k})
        #     X_k = local_param_.eval_({'view_index': k, 'data_mask': [k]})
        #     X_tilde_k = X_U_k-X_k
        #     grads = pinv(X_tilde_k.T.dot(X_tilde_k)).dot(X_tilde_k.T.dot(phi_k))
        #     Psi[:,k,:] = (local_param_.Psi[k,:].dot(grads)).T

        X_tilde = []
        N = phi.shape[1]
        n,p = X.shape
        Phi_tilde = []
        Psi = np.zeros((N, n, p))
        C_list = []
        A_list = []
        for k in range(n):
            U_k = U[k,:].indices
            phi_k = phi[U_k,:] - phi[k,:][None,:]
            X_k = X[U_k,:] - X[k,:][None,:]
            X_tilde.append(X_k.T)
            Phi_tilde.append(phi_k.T)
            
            C_k = X_k.T.dot(X_k)
            A_k = X_k.T.dot(phi_k)
            #Psi[:,k,:] = pinv(C_k).dot(A_k).T
            C_list.append(C_k)
            A_list.append(A_k)

        # OLDEST
        # print('Estimating gradients:')
        # model = Model(cov, b, L, lmbda=local_opts['reg'], device=local_opts['device'])
        # optim = torch.optim.Adam(model.parameters(), lr=local_opts['alpha'])
        # for i in range(local_opts['max_iter']):
        #     loss = model.forward()
        #     loss = loss + const_term
        #     loss = loss/(n*N)
        #     loss.backward()
        #     optim.step()
        #     optim.zero_grad()
        #     print(f"loss ({i}): {loss.item()}")

        print('Estimating gradients:')
        # model = Model(X_tilde, Phi_tilde, L, Psi=Psi, lmbda=local_opts['reg'], device=local_opts['device'])
        # optim = torch.optim.Adam(model.parameters(), lr=local_opts['alpha'])
        # for i in range(local_opts['max_iter']):
        #     loss = model.forward()
        #     loss.backward()
        #     optim.step()
        #     optim.zero_grad()
        #     print(f"loss ({i}): {loss.item()}")
        # local_subspace = model.Psi.cpu().detach().numpy()
        
        W = -L.copy().tocsr()
        W.setdiag(0)
        W1_list = []
        W2_list = []
        U_list = []
        for k in range(n):
            U_k = U[k,:].indices
            U_list.append(U_k)
            W1_list.append(np.array(W[k,U_k].todense()).flatten())
            W2_list.append(np.array(W[U_k,k].todense()).flatten())
            C_k = C_list[k].copy()
            np.fill_diagonal(C_k, C_k.diagonal() + local_opts['reg']*(np.sum(W1_list[k])+np.sum(W2_list[k])))
            C_list[k] = pinv(C_k)
        
        local_subspace = Psi.copy()
        local_subspaces = [Psi]
        is_converged = False
        np.random.seed(42)
        inds = np.arange(n)
        for i in range(local_opts['max_iter']):
            np.random.shuffle(inds)
            for k in inds.tolist():
                U_k = U_list[k]
                w_1 = W1_list[k]
                w_2 = W2_list[k]
                pinvC_k = C_list[k]
                A_k = A_list[k].copy()
                Psi_k = Psi[:,U_k,:]
                A_k += local_opts['reg']*np.sum(w_1[None,:,None]*Psi_k, axis=1).T
                A_k += local_opts['reg']*np.sum(w_2[None,:,None]*Psi_k, axis=1).T
                Psi[:,k,:] = pinvC_k.dot(A_k).T
            delta = np.sum(np.abs(local_subspace - Psi))/(n*N*p)
            if delta < local_opts['tol']:
                print('Converged at iter:', i)
                break
            print('Iter:', i+1, ':: mean of |grad_{t+1}-grad_{t}|:', delta) 
            # Psi = local_subspace.copy()
            # local_subspaces.append(Psi)
            local_subspace = Psi.copy()
            local_subspaces.append(local_subspace)
            
        self.local_subspace = local_subspace
        self.local_subspaces = local_subspaces

        # self.nnmodel = NNModel(X, phi, n_epochs=local_opts['max_iter'], lr=local_opts['alpha'])
        # local_subspace = self.nnmodel.Psi.copy()
        # self.local_subspace = local_subspace
        
        def target_proc(p_num, chunk_sz, q_):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = n
            else:
                end_ind = (p_num+1)*chunk_sz

            for k in range(start_ind, end_ind):
                U_k = U[k,:].indices
                local_param.mu[k,:] = np.mean(X[U_k,:], axis=0)

                if local_opts['explain_var'] > 0:
                    Q_k,Sigma_k,_ = svd(local_subspace[:,k,:].T)
                    var = np.cumsum(Sigma_k/np.sum(Sigma_k))
                    var_explained[k,:] = var
                    d1 = min(d, np.sum(var < local_opts['explain_var'])+1)
                else:
                    d1 = d
                    if d in local_subspace.shape:
                        Q_k,Sigma_k,_ = svd(local_subspace[:,k,:].T)
                    else:
                        Q_k,Sigma_k,_ = svds(local_subspace[:,k,:].T, d, which='LM')

                    var_explained[k,:d] = Sigma_k/np.sum(Sigma_k)
                n_pc_dir_chosen[k] = d1
                local_param.Psi[k,:,:d1] = Q_k[:,:d1]
                # Compute zeta_{kk}
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))
            
            q_.put((start_ind, end_ind,
                    local_param.zeta[start_ind:end_ind],
                    local_param.Psi[start_ind:end_ind,:],
                    local_param.mu[start_ind:end_ind,:],
                    var_explained[start_ind:end_ind,:],
                    n_pc_dir_chosen[start_ind:end_ind]))
        
        q_ = mp.Queue()
        chunk_sz = int(n/n_proc)
        proc = []
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, zeta_, Psi_, mu_, var_explained_, n_pc_dir_chosen_ = q_.get()
            local_param.zeta[start_ind:end_ind] = zeta_
            local_param.Psi[start_ind:end_ind,:] = Psi_
            local_param.mu[start_ind:end_ind,:] = mu_
            var_explained[start_ind:end_ind,:] = var_explained_
            n_pc_dir_chosen[start_ind:end_ind] = n_pc_dir_chosen_
            
        q_.close()
        
        for p_num in range(n_proc):
            proc[p_num].join()
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        local_param.var_explained = var_explained
        local_param.n_pc_dir_chosen = n_pc_dir_chosen
        return local_param
    
    def compute_RFFLE(self, d, d_e, phi, U, Atilde, gamma, local_opts, print_prop = 0.25):
        n, N = phi.shape
        N = phi.shape[1]
        if local_opts['brute_force'] and (d==2):
            all_pairs = [list(i) for i in itertools.combinations(np.arange(N).tolist(), d)]
        else:
            all_pairs = None
        tau = local_opts['tau']
        delta = local_opts['delta']
        local_param = Param('RFFLE')
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
                # Grab the precomputed U_k, Atilde_{kij}, gamma_{ki}
                U_k = U[k,:].indices
                gamma_k = gamma[k,:]
                d_e_k = d_e[np.ix_(U_k, U_k)]
                
                if all_pairs is not None:
                    zeta_min = np.inf
                    zeta_min_ind = None
                    
                    for i in all_pairs:
                        # Compute Psi_k
                        local_param.Psi_gamma[k,:] = gamma_k[i]
                        local_param.Psi_i[k,:] = i

                        # Compute zeta_{kk}
                        
                        zeta_ = compute_zeta(d_e_k, local_param.eval_({'view_index': k,
                                                                        'data_mask': U_k}))
                        if zeta_min > zeta_:
                            zeta_min = zeta_
                            zeta_min_ind = i
                     
                    local_param.Psi_gamma[k,:] = gamma_k[zeta_min_ind]
                    local_param.Psi_i[k,:] = zeta_min_ind
                    local_param.zeta[k] = zeta_min
                else:
                    # to store i_1, ..., i_d
                    i = np.zeros(d, dtype='int')
                    
                    Atilde_k = Atilde[k,:,:]

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
    
    def compute_LISOMAP(self, d, X, d_e, U, local_opts, print_prop = 0.25):
        n = U.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        
        local_param = Param('LISOMAP')
        local_param.X = X
        local_param.model = np.empty(n, dtype=object)
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
                # LPCA
                X_k = X[U_k,:]
                local_param.model[k] = Isomap(n_components=d, n_neighbors=local_opts['k_tune'])
                local_param.model[k].fit(X_k)

                # Compute zeta_{kk}
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))
            
            q_.put((start_ind, end_ind,
                    local_param.zeta[start_ind:end_ind],
                    local_param.model[start_ind:end_ind]))
        
        q_ = mp.Queue()
        chunk_sz = int(n/n_proc)
        proc = []
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, zeta_, model_ = q_.get()
            local_param.zeta[start_ind:end_ind] = zeta_
            local_param.model[start_ind:end_ind] = model_

        q_.close()
        
        for p_num in range(n_proc):
            proc[p_num].join()
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        return local_param
    
    def compute_LPCA(self, d, X, d_e, U, local_opts, print_prop = 0.25):
        n = U.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        
        local_param = Param('LPCA')
        local_param.X = X
        local_param.Psi = np.zeros((n,p,d))
        local_param.mu = np.zeros((n,p))
        local_param.zeta = np.zeros(n)
        var_explained = np.zeros((n,p))
        n_pc_dir_chosen = np.zeros(X.shape[0])
        n_proc = local_opts['n_proc']
        
        def target_proc(p_num, chunk_sz, q_):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = n
            else:
                end_ind = (p_num+1)*chunk_sz

            for k in range(start_ind, end_ind):
                U_k = U[k,:].indices
                # LPCA
                X_k = X[U_k,:]
                #print('Summary:', k, U_k.shape, flush=True)
                #print(U_k, flush=True)
                
                #print(k, flush=True)
                if local_opts['algo'] == 'L1PCA':
                    xbar_k = np.median(X_k,axis=0)[np.newaxis,:]
                    X_k = X_k - xbar_k
                    X_k = X_k.T
                    Q_k, _, _ = l1pca_optimal(X_k, d)
                elif local_opts['algo'] == 'SparsePCA':
                    xbar_k = np.median(X_k,axis=0)[np.newaxis,:]
                    X_k = X_k - xbar_k
                    alpha_ = 2
                    flag_ = True
                    while flag_ and (alpha_ > 1e-3):
                        sparse_pca = SparsePCA(n_components=d, alpha=alpha_, random_state=42)
                        sparse_pca.fit(X_k)
                        Q_k = sparse_pca.components_
                        flag_ = np.any(np.sum(Q_k**2, axis=1) < 1e-6)
                        alpha_ = 0.5*alpha_
                    
                    if flag_: # fallback to pca
                        X_k = X_k.T
                        if p == d:
                            Q_k,Sigma_k,_ = svd(X_k)
                        else:
                            Q_k,Sigma_k,_ = svds(X_k, d, which='LM')
                    else:
                        Q_k = pinv(Q_k)
                else:
                    if local_opts['algo'] == 'RPCA-GODEC':
                        power = local_opts['power']
                        lambda1 = local_opts['lambda1_init']
                        lambda1_decay = local_opts['lambda1_decay']
                        lambda1_min = local_opts['lambda1_min']
                        sparsity_frac_threshold = 1-local_opts['max_sparsity']
                        flag_ = True
                        tol = 1e-6
                        while flag_ and (lambda1 > lambda1_min):
                            X_k_hat, S_k_hat, _, _, _ = rpca_godec(X_k.T, d, power=power,
                                                                     lambda1=lambda1, tol=tol,
                                                                     random_state=42)
                            nnz_frac = np.sum(np.abs(S_k_hat)>tol)/np.prod(S_k_hat.shape)
                            # print(lambda1, nnz_frac, flush=True)
                            # print(X_k, X_k_hat.T, S_k_hat.T)
                            # print('#'*50)
                            flag_ = nnz_frac < sparsity_frac_threshold
                            lambda1 = lambda1_decay*lambda1
                            
                        X_k = X_k_hat.T
                    xbar_k = np.mean(X_k,axis=0)[np.newaxis,:]
                    X_k = X_k - xbar_k
                    X_k = X_k.T
                    if local_opts['explain_var'] > 0:
                        Q_k,Sigma_k,_ = svd(X_k)
                        var = np.cumsum(Sigma_k/np.sum(Sigma_k))
                        var_explained[k,:] = var
                        d1 = min(d, np.sum(var < local_opts['explain_var'])+1)
                    else:
                        d1 = d
                        if d in X_k.shape:
                            Q_k,Sigma_k,_ = svd(X_k)
                        else:
                            Q_k,Sigma_k,_ = svds(X_k, d, which='LM')
                        var_explained[k,:d] = Sigma_k/np.sum(Sigma_k)
                    n_pc_dir_chosen[k] = d1

                local_param.Psi[k,:,:d1] = Q_k[:,:d1]
                local_param.mu[k,:] = xbar_k

                # Compute zeta_{kk}
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))
            
            q_.put((start_ind, end_ind,
                    local_param.zeta[start_ind:end_ind],
                    local_param.Psi[start_ind:end_ind,:],
                    local_param.mu[start_ind:end_ind,:],
                    var_explained[start_ind:end_ind,:],
                    n_pc_dir_chosen[start_ind:end_ind]))
        
        q_ = mp.Queue()
        chunk_sz = int(n/n_proc)
        proc = []
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, zeta_, Psi_, mu_, var_explained_, n_pc_dir_chosen_ = q_.get()
            local_param.zeta[start_ind:end_ind] = zeta_
            local_param.Psi[start_ind:end_ind,:] = Psi_
            local_param.mu[start_ind:end_ind,:] = mu_
            var_explained[start_ind:end_ind,:] = var_explained_
            n_pc_dir_chosen[start_ind:end_ind] = n_pc_dir_chosen_

        q_.close()
        
        for p_num in range(n_proc):
            proc[p_num].join()
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        local_param.var_explained = var_explained
        local_param.n_pc_dir_chosen = n_pc_dir_chosen
        return local_param

    def compute_EXTPCA(self, d, X, d_e, U, local_opts, print_prop = 0.25):
        n = U.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        
        local_param = Param('LPCA')
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
                if d in local_opts['local_subspace'].shape:
                    Q_k,Sigma_k,_ = svd(local_opts['local_subspace'][:,k,:].T)
                else:
                    Q_k,Sigma_k,_ = svds(local_opts['local_subspace'][:,k,:].T, d, which='LM')

                local_param.Psi[k,:,:] = Q_k[:,:d]
                local_param.mu[k,:] = X[k,:]

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
    
    def compute_LKPCA(self, d, X, d_e, U, local_opts, print_prop = 0.25):
        n = U.shape[0]
        p = X.shape[1]
        print_freq = int(print_prop * n)
        
        local_param = Param('LKPCA')
        local_param.X = X
        local_param.model = np.empty(n, dtype=object)
        local_param.zeta = np.zeros(n)
        n_proc = local_opts['n_proc']
        
        def target_proc(p_num, chunk_sz, q_):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = n
            else:
                end_ind = (p_num+1)*chunk_sz

            for k in range(start_ind, end_ind):
                local_param.model[k] = KernelPCA(n_components=d, kernel=local_opts['lkpca_kernel'])
                U_k = U[k,:].indices
                X_k = X[U_k,:]
                local_param.model[k].fit(X_k)

                # Compute zeta_{kk}
                d_e_k = d_e[np.ix_(U_k, U_k)]
                local_param.zeta[k] = compute_zeta(d_e_k,
                                                   local_param.eval_({'view_index': k,
                                                                      'data_mask': U_k}))
            
            q_.put((start_ind, end_ind,
                    local_param.zeta[start_ind:end_ind],
                    local_param.model[start_ind:end_ind]))
        
        q_ = mp.Queue()
        chunk_sz = int(n/n_proc)
        proc = []
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, zeta_, model_ = q_.get()
            local_param.zeta[start_ind:end_ind] = zeta_
            local_param.model[start_ind:end_ind] = model_

        q_.close()
        
        for p_num in range(n_proc):
            proc[p_num].join()
        print('local_param: all %d points processed...' % n)
        print("max distortion is %f" % (np.max(local_param.zeta)))
        return local_param
    
    def compute_Smooth_LPCA(self, d, X, d_e, U, L, local_opts, print_prop = 0.25):
        local_param_ = self.compute_LPCA(d, X, d_e, U, local_opts)
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
        
        local_param = Param('LPCA')
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
    
    def postprocess(self, d_e, local_param_pre, U, local_opts):
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

        def target_proc(p_num, chunk_sz, barrier, U, local_param, d_e):
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
                                                             U, local_param, d_e),
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

        if local_opts['algo'] == 'LDLE' or local_opts['algo'] == 'RFFLE' or local_opts['algo'] == 'LEPC':
            local_param.Psi_i = local_param.Psi_i[npo,:]
            local_param.Psi_gamma = local_param.Psi_gamma[npo,:]
        elif local_opts['algo'] != 'LPCA' and local_opts['algo'] != 'EXTPCA':
            local_param.model = local_param.model[npo]
        else:
            local_param.Psi = local_param.Psi[npo,:]
            local_param.mu = local_param.mu[npo,:]
            
        print('Max local distortion after postprocessing:', np.max(local_param.zeta))
        return local_param