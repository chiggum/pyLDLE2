import pdb
import numpy as np
import time

from .util_ import procrustes

import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

import multiprocess as mp

# Computes Z_s for the case when to_tear is True.
# Input Z_s is the Z_s for the case when to_tear is False.
# Output Z_s is a subset of input Z_s.
def compute_Z_s_to_tear(y, s, Z_s, C, c, k):
    n_Z_s = Z_s.shape[0]
    # C_s_U_C_Z_s = (self.C[s,:]) | np.isin(self.c, Z_s)
    C_s_U_C_Z_s = (C[s,:]) | np.any(C[Z_s,:], 0)
    c = c[C_s_U_C_Z_s]
    n_ = c.shape[0]

    d_e_ = squareform(pdist(y[C_s_U_C_Z_s,:]))

    k_ = min(k,d_e_.shape[0]-1)
    neigh = NearestNeighbors(n_neighbors=k_,
                             metric='precomputed',
                             algorithm='brute')
    neigh.fit(d_e_)
    neigh_dist, _ = neigh.kneighbors()

    epsilon = neigh_dist[:,[k_-1]]
    Ug = d_e_ < (epsilon + 1e-12)

    Utildeg = np.zeros((n_Z_s+1,n_))
    for m in range(n_Z_s):
        Utildeg[m,:] = np.any(Ug[c==Z_s[m],:], 0)

    Utildeg[n_Z_s,:] = np.any(Ug[c==s,:], 0)

    # |Utildeg_{mm'}|
    n_Utildeg_Utildeg = np.matmul(Utildeg, Utildeg.T)
    np.fill_diagonal(n_Utildeg_Utildeg, 0)

    return Z_s[n_Utildeg_Utildeg[-1,:-1]>0]


def sequential_init(seq, rho, y, is_visited_view, d, Utilde, n_Utilde_Utilde,
                    C, c, intermed_param, global_opts, print_freq=1000,
                    ret_contrib_of_views=False):   
    n = Utilde.shape[1]
    if ret_contrib_of_views:
        # Initialization contribution of view i
        # as the points in cluster i
        contrib_of_view = C.copy()
    # Traverse views from 2nd view
    for m in range(1,seq.shape[0]):
        if print_freq and np.mod(m, print_freq)==0:
            print('Initial alignment of %d views completed' % m, flush=True)
        s = seq[m]
        # pth view is the parent of sth view
        p = rho[s]
        Utilde_s = Utilde[s,:]

        
        # If to tear apart closed manifolds
        if global_opts['to_tear']:
            if global_opts['init_algo_align_w_parent_only']:
                Z_s = [p]
            else:
                # Compute T_s and v_s by aligning
                # the embedding of the overlap Utilde_{sp}
                # due to sth view with that of the pth view
                Utilde_s_p = Utilde_s*Utilde[p,:]
                V_s_p = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s_p})
                V_p_s = intermed_param.eval_({'view_index': p, 'data_mask': Utilde_s_p})
                intermed_param.T[s,:,:], intermed_param.v[s,:] = procrustes(V_s_p, V_p_s)

                # Compute temporary global embedding of point in sth cluster
                C_s = C[s,:]
                y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})

                # Find more views to align sth view with
                Z_s = is_visited_view & (n_Utilde_Utilde[s,:]>0)
                Z_s_all = np.where(Z_s)[0]
                Z_s = compute_Z_s_to_tear(y, s, Z_s_all, C, c, global_opts['k'])
        # otherwise
        else:
            # Align sth view with all the views which have
            # an overlap with sth view in the ambient space
            Z_s = is_visited_view & (n_Utilde_Utilde[s,:]>0)
            Z_s = np.where(Z_s)[0]

            Z_s = Z_s.tolist()
            # If for some reason Z_s is empty
            if len(Z_s)==0:
                Z_s = [p]
                
        # Compute centroid mu_s
        # n_Utilde_s_Z_s[k] = #views in Z_s which contain
        # kth point if kth point is in the sth view, else zero
        n_Utilde_s_Z_s = np.zeros(n, dtype=int)
        mu_s = np.zeros((n,d))
        for mp in Z_s:
            Utilde_s_mp = Utilde_s & Utilde[mp,:]
            if ret_contrib_of_views:
                #overlaps[s,:] = overlaps[s,:] | Utilde_s_mp
                #overlaps[mp,:] = overlaps[mp,:] | (Utilde[mp,:] & C[s,:])
                contrib_of_view[s,:] = contrib_of_view[s,:] | (Utilde_s & C[mp,:])
                
            n_Utilde_s_Z_s[Utilde_s_mp] += 1
            mu_s[Utilde_s_mp,:] += intermed_param.eval_({'view_index': mp, 'data_mask': Utilde_s_mp})

        # Compute T_s and v_s by aligning the embedding of the overlap
        # between sth view and the views in Z_s, with the centroid mu_s
        temp = n_Utilde_s_Z_s > 0
        mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]
        V_s_Z_s = intermed_param.eval_({'view_index': s, 'data_mask': temp})

        T_s, v_s = procrustes(V_s_Z_s, mu_s)

        # Update T_s, v_
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s

        # Mark sth view as visited
        is_visited_view[s] = True

        # Compute global embedding of point in sth cluster
        C_s = C[s,:]
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    
    if ret_contrib_of_views:
        return y, is_visited_view, contrib_of_view
    else:
        return y, is_visited_view


# Ngoc-Diep Ho, Paul Van Dooren, On the pseudo-inverse of the Laplacian of a bipartite graph
def compute_Lpinv(Utilde):
    B_ = Utilde.T.astype('int')
    D_1 = np.sum(B_, axis=1, keepdims=True)
    D_2 = np.sum(B_, axis=0, keepdims=True)
    D_1_inv_sqrt = np.sqrt(1/D_1)
    D_2_inv_sqrt = np.sqrt(1/D_2)
    B_tilde = D_1_inv_sqrt*B_*D_2_inv_sqrt
    U12,SS,VT = scipy.linalg.svd(B_tilde, full_matrices=False)
    V = VT.T
    mask = np.abs(SS-1)<1e-6
    m_1 = np.sum(mask)
    Sigma = np.expand_dims(SS[m_1:], 1)
    Sigma_1 = 1/(1-Sigma**2)
    Sigma_2 = Sigma*Sigma_1
    U1 = U12[:,:m_1]
    U2 = U12[:,m_1:]
    V1 = V[:,:m_1]
    V2 = V[:,m_1:]
    
    def target_proc(i):
        if i == 0:
            tmp = -0.75*np.matmul(U1,U1.T) + np.matmul(U2, ((Sigma_1-1))*(U2.T)) + np.eye(U12.shape[0])
            tmp = tmp*D_1_inv_sqrt*(D_1_inv_sqrt.T)
        elif i==1:
            tmp = -0.25*np.matmul(U1,V1.T) + np.matmul(U2, Sigma_2*(V2.T))
            tmp = tmp*D_1_inv_sqrt*D_2_inv_sqrt
        elif i==2:
            tmp = 0.25*np.matmul(V1,V1.T) + np.matmul(V2, Sigma_1*(V2.T))
            tmp = tmp*(D_2_inv_sqrt.T)*D_2_inv_sqrt
        return tmp, np.sum(tmp, axis=1, keepdims=True), np.sum(tmp, axis=0, keepdims=True)
    
    L_tilde = []
    L_tilde_sum_1 = []
    L_tilde_sum_0 = []
    ################################################
    # Parallel
#     with mp.Pool(3) as p:
#         tmp = p.map(target_proc, [0,1,2])
#         for elem in tmp:
#             L_tilde.append(elem[0])
#             L_tilde_sum_1.append(elem[1])
#             L_tilde_sum_0.append(elem[2])
        
    # Sequential
    for i in range(3):
        elem = target_proc(i)
        L_tilde.append(elem[0])
        L_tilde_sum_1.append(elem[1])
        L_tilde_sum_0.append(elem[2])
    ################################################
    
    L_pinv_1 = np.concatenate((L_tilde[0], L_tilde[1]), axis=1)
    L_pinv_2 = np.concatenate((L_tilde[1].T, L_tilde[2]), axis=1)
    Lpinv = np.concatenate([L_pinv_1, L_pinv_2], axis=0)

    n1 = Lpinv.shape[1]
    v1 = np.concatenate((L_tilde_sum_1[0]+L_tilde_sum_1[1],
                         L_tilde_sum_0[1].T+L_tilde_sum_1[2]), axis=0)
    n0 = Lpinv.shape[0]
    v0 = np.concatenate((L_tilde_sum_0[0]+L_tilde_sum_1[1].T,
                         L_tilde_sum_0[1]+L_tilde_sum_0[2]), axis=1)
    Lpinv = Lpinv - v1/n1 - v0.T/n0
    return Lpinv
    

def compute_CC(D, B, Lpinv):
    CC = D - np.matmul(B, np.matmul(Lpinv, B.T))
    CC = 0.5*(CC + CC.T)
    return CC

    
def build_ortho_optim(d, Utilde, intermed_param, tol = 1e-6, Lpinv=None):
    M,n = Utilde.shape
    B = np.zeros((M*d,n+M))
    D = np.zeros((M*d,M*d))
    for i in range(M):
        X_ = intermed_param.eval_({'view_index': i,
                                   'data_mask': Utilde[i,:]})
        D[d*i:d*(i+1),d*i:d*(i+1)] = np.matmul(X_.T,X_)
        B[d*i:d*(i+1),n+i] = -np.sum(X_.T, axis=1)
        B[d*i:d*(i+1),np.where(Utilde[i,:])[0]] = X_.T

    
    if Lpinv is None:
        print('Computing Pseudoinverse of a matrix of L of size', n+M, flush=True)
        Lpinv = compute_Lpinv(Utilde)
    
    CC = compute_CC(D, B, Lpinv)
    return CC, Lpinv, B
    

def compute_alignment_err(d, Utilde, intermed_param, tol = 1e-6, CC=None, Lpinv=None, B=None):
    if (CC is None) or (Lpinv is None) or (B is None):
        CC, Lpinv, B = build_ortho_optim(d, Utilde, intermed_param, tol = 1e-6)
    else:
        CC, Lpinv, B = build_ortho_optim(d, Utilde, intermed_param, tol = 1e-6, Lpinv=Lpinv)
    err = 0
    M,n = Utilde.shape
    CC_mask = np.tile(np.eye(d, dtype=bool), (M,M))
    err = np.sum(CC[CC_mask])
    return err, [CC, Lpinv, B]
    
def spectral_init(y, is_visited_view, d, Utilde,
                  C, intermed_param, global_opts, print_freq=1000,
                  tol = 1e-6):
    CC, Lpinv, B = build_ortho_optim(d, Utilde, intermed_param, tol = 1e-6)
    M,n = Utilde.shape
    print('Computing eigh(C,k=d)', flush=True)
    np.random.seed(42)
    v0 = np.random.uniform(0,1,CC.shape[0])
    # To find smallest eigenvalues, using shift-inverted algo with mode=normal and which='LM'
    W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, sigma=0.0)
    print('Done.', flush=True)
    Wstar = np.sqrt(M)*V_.T
    
    Tstar = np.zeros((d, M*d))
    for i in range(M):
        U_,S_,VT_ = scipy.linalg.svd(Wstar[:,d*i:d*(i+1)])
        temp_ = np.matmul(U_,VT_)
        if (global_opts['init_algo_name'] != 'spectral') and (np.linalg.det(temp_) < 0):
            VT_[-1,:] *= -1
            Tstar[:,i*d:(i+1)*d] = np.matmul(U_, VT_)
        else:
            Tstar[:,i*d:(i+1)*d] = temp_
    
    Zstar = np.matmul(Tstar, np.matmul(B, Lpinv))
    
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:]
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
        is_visited_view[s] = 1
    
    return y, Zstar[:,:n].T, is_visited_view

def sequential_final(y, d, Utilde, C, intermed_param, n_Utilde_Utilde, n_Utildeg_Utildeg,
                     first_intermed_view_in_cluster,
                     parents_of_intermed_views_in_cluster, cluster_of_intermed_view,
                     global_opts):
    M,n = Utilde.shape
    # Traverse over intermediate views in a random order
    seq = np.random.permutation(M)

    # For a given seq, refine the global embedding
    for it1 in range(global_opts['refine_algo_max_internal_iter']):
        for s in seq.tolist():
            # Never refine s_0th intermediate view
            if s in first_intermed_view_in_cluster:
                C_s = C[s,:]
                y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
                continue

            Utilde_s = Utilde[s,:]

            # If to tear apart closed manifolds
            if global_opts['to_tear']:
                # Find more views to align sth view with
                Z_s = (n_Utilde_Utilde[s,:] > 0) & (n_Utildeg_Utildeg[s,:] > 0)
            # otherwise
            else:
                # Align sth view with all the views which have
                # an overlap with sth view in the ambient space
                Z_s = n_Utilde_Utilde[s,:] > 0

            Z_s = np.where(Z_s)[0].tolist()

            if len(Z_s) == 0:
                Z_s = parents_of_intermed_views_in_cluster[cluster_of_intermed_view[s]][s]
                Z_s = [Z_s]

            # Compute centroid mu_s
            # n_Utilde_s_Z_s[k] = #views in Z_s which contain
            # kth point if kth point is in the sth view, else zero
            n_Utilde_s_Z_s = np.zeros(n, dtype=int)
            mu_s = np.zeros((n,d))
            for mp in Z_s:
                Utilde_s_mp = Utilde_s & Utilde[mp,:]
                n_Utilde_s_Z_s[Utilde_s_mp] += 1
                mu_s[Utilde_s_mp,:] += intermed_param.eval_({'view_index': mp, 'data_mask': Utilde_s_mp})

            temp = n_Utilde_s_Z_s > 0
            mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]

            # Compute T_s and v_s by aligning the embedding of the overlap
            # between sth view and the views in Z_s, with the centroid mu_s
            V_s_Z_s = intermed_param.eval_({'view_index': s, 'data_mask': temp})
            
            T_s, v_s = procrustes(V_s_Z_s, mu_s)

            # Update T_s, v_s
            intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
            intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s

            # Compute global embedding of points in sth cluster
            C_s = C[s,:]
            y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y

def retraction_final(y, d, Utilde, C, intermed_param,
                     n_Utilde_Utilde, n_Utildeg_Utildeg,
                     first_intermed_view_in_cluster,
                     parents_of_intermed_views_in_cluster,
                     cluster_of_intermed_view,
                     global_opts, CC=None, Lpinv=None, B=None):
    if (CC is None) or (Lpinv is None) or (B is None):
        CC, Lpinv, B = build_ortho_optim(d, Utilde, intermed_param, tol = 1e-6)
    else:
        CC, Lpinv, B = build_ortho_optim(d, Utilde, intermed_param, tol = 1e-6, Lpinv=Lpinv)
    M,n = Utilde.shape
    n_proc = min(M,global_opts['n_proc'])
    def skew(A):
        return 0.5*(A-A.T)

    def unique_qr(A):
        Q, R = np.linalg.qr(A)
        signs = 2 * (np.diag(R) >= 0) - 1
        Q = Q * signs[np.newaxis, :]
        R = R * signs[:, np.newaxis]
        return Q, R

    def update(O, t):
        O = np.copy(O)
        xi = 2*np.matmul(O, CC)
        I_d = np.eye(d)
        
        ###########################################
        # Parallel Updates
        ###########################################
        def target_proc(p_num, chunk_sz, q_):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = M
            else:
                end_ind = (p_num+1)*chunk_sz
            for i in range(start_ind, end_ind):
                temp0 = O[:,i*d:(i+1)*d]
                temp1 = skew(np.matmul(xi[:,i*d:(i+1)*d],temp0.T))
                Q_,R_ = unique_qr(temp0 - t*np.matmul(temp1,temp0))
                O[:,i*d:(i+1)*d] = Q_

            q_.put((start_ind, end_ind, O[:,start_ind*d:end_ind*d]))
        
        q_ = mp.Queue()
        proc = []
        chunk_sz = int(M/n_proc)
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz,q_),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            start_ind, end_ind, O_ = q_.get()
            O[:,start_ind*d:end_ind*d] = O_

        q_.close()
        for p_num in range(n_proc):
            proc[p_num].join()
        ###########################################
        
        # Sequential version of above
        # for i in range(M):
        #     temp0 = O[:,i*d:(i+1)*d]
        #     temp1 = skew(np.matmul(xi[:,i*d:(i+1)*d],temp0.T))
        #     Q_,R_ = unique_qr(temp0 - t*np.matmul(temp1,temp0))
        #     O[:,i*d:(i+1)*d] = Q_
        
        return O

    alpha = global_opts['refine_algo_alpha']
    max_iter = global_opts['refine_algo_max_internal_iter']
    Tstar = np.zeros((d,M*d))
    for s in range(M):
        Tstar[:,s*d:(s+1)*d] = np.eye(d)
    
    print('Descent starts', flush=True)
    for _ in range(max_iter):
        Tstar = update(Tstar, alpha)
    
    Zstar = np.matmul(Tstar, np.matmul(B, Lpinv))
    
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:]
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})

    return y, [CC, Lpinv, B]