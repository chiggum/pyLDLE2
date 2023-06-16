import pdb
import numpy as np
import time

from .util_ import procrustes, issparse, sparse_matrix, nearest_neighbors

import scipy
from scipy.sparse import linalg as slinalg
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, block_diag, vstack, bmat
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra

import multiprocess as mp
from multiprocess import shared_memory

from sklearn.manifold._locally_linear import null_space

import scs

def compute_far_off_points(d_e, global_opts, force_compute=False):
    if 'reuse' in global_opts['far_off_points_type'] and (not force_compute):
        return global_opts['far_off_points']

    if global_opts['far_off_points_type'] != 'random':
        np.random.seed(42)
    far_off_points = []
    dist_from_far_off_points = None
    while len(far_off_points) < global_opts['n_repel']:
        if len(far_off_points) == 0:
            far_off_points = [np.random.randint(0,d_e.shape[0])]
            dist_from_far_off_points = dijkstra(d_e, directed=False,
                                                indices=far_off_points[-1])
        else:
            far_off_points.append(np.argmax(dist_from_far_off_points))
            dist_from_far_off_points = np.minimum(dist_from_far_off_points,
                                                  dijkstra(d_e, directed=False,
                                                           indices=far_off_points[-1]))
    return far_off_points
    

# Computes Z_s for the case when to_tear is True.
# Input Z_s is the Z_s for the case when to_tear is False.
# Output Z_s is a subset of input Z_s.
def compute_Z_s_to_tear(y, s, Z_s, C, c, k):
    n_Z_s = Z_s.shape[0]
    # C_s_U_C_Z_s = (self.C[s,:]) | np.isin(self.c, Z_s)
    C_s_U_C_Z_s = np.where(C[s,:] + C[Z_s,:].sum(axis=0))[1]
    n_ = C_s_U_C_Z_s.shape[0]
    k_ = min(k,n_-1)
    _, neigh_ind_ = nearest_neighbors(y[C_s_U_C_Z_s,:], k_, 'euclidean')
    U_ = sparse_matrix(neigh_ind_, np.ones(neigh_ind_.shape, dtype=bool))
    Utilde_ = C[np.ix_(Z_s,C_s_U_C_Z_s)].dot(U_)
    Utilde_ = vstack([Utilde_, C[s,C_s_U_C_Z_s].dot(U_)])
    n_Utildeg_Utilde_ = Utilde_.dot(Utilde_.T) 
    n_Utildeg_Utilde_.setdiag(False)
    return Z_s[n_Utildeg_Utilde_[-1,:-1].nonzero()[1]].tolist()

def procrustes_init(seq, rho, y, is_visited_view, d, Utilde, n_Utilde_Utilde,
                    C, c, intermed_param, global_opts, print_freq=1000):   
    n = Utilde.shape[1]
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
            if global_opts['align_w_parent_only']:
                Z_s = [p]
            else:
                # Compute T_s and v_s by aligning
                # the embedding of the overlap Utilde_{sp}
                # due to sth view with that of the pth view
                Utilde_s_p = Utilde_s.multiply(Utilde[p,:]).nonzero()[1]
                V_s_p = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s_p})
                V_p_s = intermed_param.eval_({'view_index': p, 'data_mask': Utilde_s_p})
                intermed_param.T[s,:,:], intermed_param.v[s,:] = procrustes(V_s_p, V_p_s)
                
                # Compute temporary global embedding of point in sth cluster
                C_s = C[s,:].indices
                y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
                # Find more views to align sth view with
                Z_s = n_Utilde_Utilde[s,:].multiply(is_visited_view)
                Z_s_all = Z_s.nonzero()[1]
                Z_s = compute_Z_s_to_tear(y, s, Z_s_all, C, c, global_opts['k'])
                # The parent must be in Z_s
                if p not in Z_s:
                    Z_s.append(p)
        # otherwise
        else:
            # Align sth view with all the views which have
            # an overlap with sth view in the ambient space
            Z_s = n_Utilde_Utilde[s,:].multiply(is_visited_view)
            Z_s = Z_s.nonzero()[1].tolist()
            # If for some reason Z_s is empty
            if len(Z_s)==0:
                Z_s = [p]
                
        # Compute centroid mu_s
        # n_Utilde_s_Z_s[k] = #views in Z_s which contain
        # kth point if kth point is in the sth view, else zero
        n_Utilde_s_Z_s = np.zeros(n, dtype=int)
        mu_s = np.zeros((n,d))
        cov_s = csr_matrix((1,n), dtype=bool)
        for mp in Z_s:
            Utilde_s_mp = Utilde_s.multiply(Utilde[mp,:]).nonzero()[1]    
            n_Utilde_s_Z_s[Utilde_s_mp] += 1
            mu_s[Utilde_s_mp,:] += intermed_param.eval_({'view_index': mp,
                                                         'data_mask': Utilde_s_mp})

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
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, is_visited_view

def compute_Lpinv_helpers(W):
    M, n = W.shape
    # B_ = W.copy().transpose().astype('int')
    B_ = W.copy().transpose().astype('float')
    D_1 = np.asarray(B_.sum(axis=1))
    D_2 = np.asarray(B_.sum(axis=0))
    D_1_inv_sqrt = np.sqrt(1/D_1)
    D_2_inv_sqrt = np.sqrt(1/D_2)
    B_tilde = B_.multiply(D_2_inv_sqrt).multiply(D_1_inv_sqrt)
    # TODO: U12 is dense of size nxM
    print('Computing svd', flush=True)
    U12,SS,VT = scipy.linalg.svd(B_tilde.todense(), full_matrices=False)
    print('Done', flush=True)
    # U12,SS,VT = slinalg.svds(B_tilde, k=M, solver='propack')
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
    return [D_1_inv_sqrt, D_2_inv_sqrt, U1, U2, V1, V2, Sigma_1, Sigma_2]

def compute_Lpinv_MT(Lpinv_helpers, B):
    D_1_inv_sqrt, D_2_inv_sqrt, U1, U2, V1, V2, Sigma_1, Sigma_2 = Lpinv_helpers
    n = D_1_inv_sqrt.shape[0]
    B_mean = B.mean(axis=1)
    if len(B_mean.shape) == 1:
        B_mean = B_mean[:,None]
    B_n = B - B_mean
    B_n = np.asarray(B_n)
    B1T = D_1_inv_sqrt * (B_n[:, :n].T)
    B2T = D_2_inv_sqrt.T * (B_n[:, n:].T)
    
    U1TB1T = np.matmul(U1.T, B1T)
    U2TB1T = np.matmul(U2.T, B1T)
    V1TB2T = np.matmul(V1.T, B2T)
    V2TB2T = np.matmul(V2.T, B2T)
    
    temp1 = -0.75*np.matmul(U1,U1TB1T)-0.25*np.matmul(U1,V1TB2T) +\
            np.matmul(U2, ((Sigma_1-1))*(U2TB1T)) + np.matmul(U2, Sigma_2*(V2TB2T)) + B1T
    temp1 = temp1 * D_1_inv_sqrt
    
    temp2 = -0.25*np.matmul(V1, U1TB1T) + 0.25*np.matmul(V1,V1TB2T) +\
            np.matmul(V2, Sigma_2*(U2TB1T)) + np.matmul(V2, Sigma_1*(V2TB2T))
    temp2 = temp2 * D_2_inv_sqrt.T 
    
    temp = np.concatenate((temp1, temp2), axis=0)
    temp = temp - np.mean(temp, axis=0, keepdims=True)
    return temp

# Ngoc-Diep Ho, Paul Van Dooren, On the pseudo-inverse of the Laplacian of a bipartite graph
def compute_Lpinv_BT(W, B):
    D_1_inv_sqrt, D_2_inv_sqrt, U1, U2, V1, V2, Sigma_1, Sigma_2 = compute_Lpinv_helpers(W)
    return compute_Lpinv_MT(Lpinv_helpers, B)

# conjugate gradient based approach
def compute_Lpinv_BT_cg(Utilde, B):
    pdb.set_trace()
    L_0 = bmat([[None, -Utilde.T.astype(int)],[-Utilde.astype(int), None]], format='csr')
    L_0.setdiag(-np.array(np.sum(L_0,axis=1)).flatten())
    x = []
    for i in range(B.shape[0]):
        x_, info_ = slinalg.cg(L_0, B[i,:].toarray().flatten())
        if info_:
            print('CG did not converge.')
            raise
        x.append(x_[:,None])
    return np.concatenate(x, axis=1)

def compute_CC(D, B, Lpinv_BT):
    CC = D - B.dot(Lpinv_BT)
    return 0.5*(CC + CC.T)

def build_ortho_optim(d, Utilde, intermed_param,
                      far_off_points=[], repel_by=0.,
                      beta=None):
    M,n = Utilde.shape
    B_row_inds = []
    B_col_inds = []
    B_vals = []
    D = []
    
    if (beta is not None) and (beta['align'] is not None):
        W_row_inds = []
        W_col_inds = []
        W_vals = []
        for i in range(M):
            Utilde_i = Utilde[i,:].indices
            w_ki = intermed_param.alignment_wts({'view_index': i,
                                                  'data_mask': Utilde_i,
                                                  'beta': beta['align']})
            col_inds = Utilde_i.tolist()
            W_row_inds += [i]*len(col_inds)
            W_col_inds += col_inds
            W_vals += w_ki.tolist()
        W_row_inds = np.array(W_row_inds)
        W_col_inds = np.array(W_col_inds)
        W_vals = np.array(W_vals)
        for k in range(n):
            mask = W_col_inds == k
            temp = W_vals[mask]
            temp = np.exp(temp - temp.max())
            temp *= temp.shape[0]/np.sum(temp)
            W_vals[mask] = temp
        W = csr_matrix((W_vals, (W_row_inds, W_col_inds)), shape=(M,n), dtype=float)
    else:
        W = Utilde.astype(float)
        W_vals = W.data
        
    
    for i in range(M):
        Utilde_i = Utilde[i,:].indices
        X_ = intermed_param.eval_({'view_index': i,
                                   'data_mask': Utilde_i})
        sqrt_p_ki = np.sqrt(np.array(W[i,:].data).flatten()[:,None])
        X_ = sqrt_p_ki * X_
        D.append(np.matmul(X_.T,X_))
        
        row_inds = list(range(d*i,d*(i+1)))
        col_inds = Utilde_i.tolist()
        
        B_row_inds += (row_inds + np.repeat(row_inds, len(col_inds)).tolist())
        B_col_inds += (np.repeat([n+i], d).tolist() + np.tile(col_inds, d).tolist())
        
        X_ = sqrt_p_ki * X_
        B_vals += (np.sum(-X_.T, axis=1).tolist() + X_.T.flatten().tolist())
    
    D = block_diag(D, format='csr')
    B = csr_matrix((B_vals, (B_row_inds, B_col_inds)), shape=(M*d,n+M))
    
    print('min and max weights:', np.array(W_vals).min(), np.array(W_vals).max())

    n_repel = len(far_off_points)
    print('Computing Pseudoinverse of a matrix of L of size', n, '+', M, 'multiplied with B', flush=True)
    Lpinv_helpers = compute_Lpinv_helpers(W)
    
    Lpinv_BT = compute_Lpinv_MT(Lpinv_helpers, B)
    CC = compute_CC(D, B, Lpinv_BT)
    if n_repel > 0:
        temp_arr = (-np.ones(n_repel)).tolist()
        L_r = np.zeros((n_repel, n_repel))
        L__row_inds = []
        L__col_inds = []
        L__vals = []
        for i in range(n_repel):
            L__row_inds += [far_off_points[i]]*n_repel
            L__col_inds += far_off_points
            if (beta is not None) and (beta['repel'] is not None):
                p_llp = intermed_param.repulsion_wts({'pt_index': far_off_points[i],
                                                      'repelling_pts_indices': far_off_points,
                                                      'beta': beta['repel']})
                p_llp[i] = 0
                p_llp_sum = np.sum(p_llp)
                p_llp *= -1
                p_llp[i] = p_llp_sum
                L_r[i,:] = p_llp
            else:
                temp_arr[i] = n_repel-1
                L_r[i,:] = temp_arr
                temp_arr[i] = -1
            L__vals += L_r[i,:].tolist()
        
        L_ = csr_matrix((L__vals, (L__row_inds, L__col_inds)), shape=(n+M,n+M))
        L_ = -repel_by*L_
        L__Lpinv_BT = L_.dot(Lpinv_BT)
        CC = CC + (Lpinv_BT.T).dot(L__Lpinv_BT)

    return CC, Lpinv_BT, D, B
    
# unscaled alignment error
def compute_alignment_err(d, Utilde, intermed_param, scale_num, far_off_points=[], repel_by=0., beta=None):
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=far_off_points,
                                     repel_by=repel_by, beta=beta)
    M,n = Utilde.shape
    
    ## Check if C is pd or psd
    #np.random.seed(42)
    #v0 = np.random.uniform(0,1,CC.shape[0])
    #sigma_min_C = slinalg.eigsh(CC, k=1, v0=v0, which='SM',return_eigenvectors=False)
    #print('Smallest singular value of C', sigma_min_C, flush=True)
    
    CC_mask = np.tile(np.eye(d, dtype=bool), (M,M))
    #scale_denom = Utilde.sum()
    #scale = (scale_num/scale_denom)
    scale = 1
    err = np.sum(CC[CC_mask]) * scale
    return err

# Kunal N Chaudhury, Yuehaw Khoo, and Amit Singer, Global registration
# of multiple point clouds using semidefinite programming
def spectral_alignment(y, d, Utilde,
                      C, intermed_param, global_opts, 
                      seq_of_intermed_views_in_cluster):
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=global_opts['far_off_points'],
                                     repel_by=global_opts['repel_by'],
                                     beta=global_opts['beta'])
        
    M,n = Utilde.shape
    n_clusters = len(seq_of_intermed_views_in_cluster)
    CC0 = np.zeros((M*d,d))
    # for s in range(M):
    #     CC0[s*d:(s+1)*d,:] = intermed_param.T[s,:,:]
    for s in range(M):
        CC0[s*d:(s+1)*d,:] = np.eye(d)
    CC0 = CC0/np.sqrt(M)
    v0 = CC0[:,0]
    
    print('Computing eigh(C,k=d)', flush=True)
    #np.random.seed(42)
    #v0 = np.random.uniform(0,1,CC.shape[0])
    if (global_opts['n_repel'] == 0) or (global_opts['repel_by'] == 0):
        # To find smallest eigenvalues, using shift-inverted algo with mode=normal and which='LM'
        W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, sigma=-1e-3)
        #print(W_, flush=True)
        
        # W_,V_ = scipy.sparse.linalg.lobpcg(CC, CC0.T, largest=False)
        # or just pass which='SM' without using sigma
        # W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, which='SM')
    else:
        # To find smallest eigenvalues, using shift-inverted algo with mode=normal and which='LM'
        CC_frob = np.linalg.norm(CC)
        W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, sigma=-2*CC_frob)
        # or just pass which='SM' without using sigma
        # W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, which='SA')
    print('Done.', flush=True)
    Wstar = np.sqrt(M)*V_.T
    Tstar = np.zeros((d, M*d))
    
    for i in range(n_clusters):
        # the first view in each cluster is not rotated
        seq = seq_of_intermed_views_in_cluster[i]
        s0 = seq[0]
        U_,S_,VT_ = scipy.linalg.svd(Wstar[:,d*s0:d*(s0+1)])
        Q =  np.matmul(U_,VT_)
        if (global_opts['init_algo_name'] != 'spectral') and (np.linalg.det(Q) < 0): # remove reflection
            VT_[-1,:] *= -1
            Q = np.matmul(U_, VT_)
        Q = Q.T
        
        for m in range(M):
            U_,S_,VT_ = scipy.linalg.svd(Wstar[:,d*m:d*(m+1)])
            temp_ = np.matmul(U_,VT_)
            if (global_opts['init_algo_name'] != 'spectral') and (np.linalg.det(temp_) < 0): # remove reflection
                VT_[-1,:] *= -1
                temp_ = np.matmul(U_, VT_)
            Tstar[:,m*d:(m+1)*d] = np.matmul(temp_, Q)
    
    Zstar = Tstar.dot(Lpinv_BT.transpose())
    
    # the first view in each cluster is not translated
    for i in range(n_clusters):
        seq = seq_of_intermed_views_in_cluster[i]
        s0 = seq[0]
        temp = Zstar[:,n+s0][:,None].copy()
        Zstar[:,n+seq] -= temp
        C_seq = C[seq,:].sum(axis=0).nonzero()[1]
        Zstar[:,C_seq] -= temp
        
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, Zstar[:,:n].T

def ltsa_alignment(y, d, Utilde,
                  C, intermed_param, global_opts, 
                  seq_of_intermed_views_in_cluster):
    M,n = Utilde.shape
    Theta_mean = []
    Theta_pinv_normalized = []
    CC = np.zeros((n,n))
    for i in range(M):
        Utilde_i = Utilde[i,:].indices
        Theta_i = intermed_param.eval_({'view_index': i, 'data_mask': Utilde_i}).T
        Theta_mean.append(np.mean(Theta_i, axis=1))
        Theta_i = Theta_i - Theta_mean[-1][:,None]
        Theta_i_pinv = scipy.linalg.pinv(Theta_i)
        Theta_i_pinv = Theta_i_pinv - np.mean(Theta_i_pinv, axis=0)[None,:]
        W_i = -1/len(Utilde_i) - Theta_i_pinv.dot(Theta_i)
        W_i[np.diag_indices_from(W_i)] += 1
        
        CC[np.ix_(Utilde_i, Utilde_i)] += W_i
        Theta_pinv_normalized.append(Theta_i_pinv)
    
    np.random.seed(42)
    v0 = np.random.uniform(0,1,CC.shape[0])
    if (global_opts['n_repel'] == 0) or (global_opts['repel_by'] == 0):
        # To find smallest eigenvalues, using shift-inverted algo with mode=normal and which='LM'
        # W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d+1, v0=v0, sigma=-1e-3)
        # or just pass which='SM' without using sigma
        # W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, which='SM')
        
        Zstar, _ = null_space(CC, d, k_skip=1, eigen_solver='auto', random_state=42)
    else:
        print('Not implemented yet.', flush=True)
        
    for s in range(M):
        intermed_param.v[s,:] -= Theta_mean[s]
        Utilde_s = Utilde[s,:].indices
        Y_s = Zstar[Utilde_s,:].T
        T_s = np.matmul(Y_s, Theta_pinv_normalized[s]).T
        v_s = np.mean(Y_s, axis=1)[None,:]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, Zstar

def procrustes_final(y, d, Utilde, C, intermed_param,
                     seq_of_intermed_views_in_cluster, global_opts):
    M,n = Utilde.shape
    # Traverse over intermediate views in a random order
    seq = np.random.permutation(M)
    is_first_view_in_cluster = np.zeros(M, dtype=bool)
    # is_first_view_in_cluster[i] = True if the ith view is the first
    # view in some cluster of views
    for i in range(len(seq_of_intermed_views_in_cluster)):
        is_first_view_in_cluster[seq_of_intermed_views_in_cluster[i][0]] = True
        
    y_due_to_all_views = []
    for k in range(n):
        y_due_to_all_views.append({})
        
    for s in range(M):
        Utilde_s = Utilde[s,:].nonzero()[1]
        y_Utilde_s = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s})
        for k in range(len(Utilde_s)):
            y_due_to_all_views[Utilde_s[k]][s] = y_Utilde_s[k,:]

    # For a given seq, refine the global embedding
    for it1 in range(global_opts['max_internal_iter']):
        for s in seq.tolist():
            # # Never refine s_0th intermediate view
            # if is_first_view_in_cluster[s]:
            #     continue

            Utilde_s = Utilde[s,:].nonzero()[1]
            mu_s = []
            Utilde_s_ = []
            for k_ in range(len(Utilde_s)):
                k = Utilde_s[k_]
                if len(y_due_to_all_views[k]) == 1:
                    continue
                Utilde_s_.append(k)
                mu = np.array(list(y_due_to_all_views[k].values())).sum(axis=0)
                mu = (mu - y_due_to_all_views[k][s])/(len(y_due_to_all_views[k])-1)
                mu_s.append(mu)
            mu_s = np.array(mu_s)
            Utilde_s_ = np.array(Utilde_s_)

            # Compute T_s and v_s by aligning the embedding of the overlap
            # between sth view and the views in Z_s, with the centroid mu_s
            V_s_Z_s = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s_})
            T_s, v_s = procrustes(V_s_Z_s, mu_s)

            # Update T_s, v_s
            intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
            intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s

            y_Utilde_s = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s})
            for k in range(len(Utilde_s)):
                y_due_to_all_views[Utilde_s[k]][s] = y_Utilde_s[k,:]
    
    y = np.zeros((n,d))
    for k in range(n):
        y[k,:] = np.array(list(y_due_to_all_views[k].values())).mean(axis=0)
    return y

def rgd_alignment(y, d, Utilde, C, intermed_param, global_opts):
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=global_opts['far_off_points'],
                                     repel_by=global_opts['repel_by'],
                                     beta=global_opts['beta'])
    M,n = Utilde.shape
    n_proc = min(M,global_opts['n_proc'])
    barrier = mp.Barrier(n_proc)

    def update(alpha, max_iter, shm_name_O, O_shape, O_dtype,
               shm_name_CC, CC_shape, CC_dtype, barrier):
        ###########################################
        # Parallel Updates
        ###########################################
        def target_proc(p_num, chunk_sz, barrier):
            existing_shm_O = shared_memory.SharedMemory(name=shm_name_O)
            O = np.ndarray(O_shape, dtype=O_dtype, buffer=existing_shm_O.buf)
            existing_shm_CC = shared_memory.SharedMemory(name=shm_name_CC)
            CC = np.ndarray(CC_shape, dtype=CC_dtype, buffer=existing_shm_CC.buf)

            def unique_qr(A):
                Q, R = np.linalg.qr(A)
                signs = 2 * (np.diag(R) >= 0) - 1
                Q = Q * signs[np.newaxis, :]
                R = R * signs[:, np.newaxis]
                return Q, R
            
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = M
            else:
                end_ind = (p_num+1)*chunk_sz
            for _ in range(max_iter):
                O_copy = O.copy()
                barrier.wait()
                for i in range(start_ind, end_ind):
                    xi_ = 2*np.matmul(O_copy, CC[:,i*d:(i+1)*d])
                    temp0 = O[:,i*d:(i+1)*d]
                    temp1 = np.matmul(xi_,temp0.T)
                    skew_temp1 = 0.5*(temp1-temp1.T)
                    Q_,R_ = unique_qr(temp0 - alpha*np.matmul(skew_temp1,temp0))
                    O[:,i*d:(i+1)*d] = Q_
                barrier.wait()
            
            existing_shm_O.close()
            existing_shm_CC.close()
        
        
        proc = []
        chunk_sz = int(M/n_proc)
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz, barrier),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            proc[p_num].join()
        ###########################################
        
        # Sequential version of above
        # for i in range(M):
        #     temp0 = O[:,i*d:(i+1)*d]
        #     temp1 = skew(np.matmul(xi[:,i*d:(i+1)*d],temp0.T))
        #     Q_,R_ = unique_qr(temp0 - t*np.matmul(temp1,temp0))
        #     O[:,i*d:(i+1)*d] = Q_

    alpha = global_opts['alpha']
    max_iter = global_opts['max_internal_iter']
    Tstar = np.zeros((d,M*d))
    for s in range(M):
        Tstar[:,s*d:(s+1)*d] = np.eye(d)
    
    print('Descent starts', flush=True)
    shm_Tstar = shared_memory.SharedMemory(create=True, size=Tstar.nbytes)
    np_Tstar = np.ndarray(Tstar.shape, dtype=Tstar.dtype, buffer=shm_Tstar.buf)
    np_Tstar[:] = Tstar[:]
    shm_CC = shared_memory.SharedMemory(create=True, size=CC.nbytes)
    np_CC = np.ndarray(CC.shape, dtype=CC.dtype, buffer=shm_CC.buf)
    np_CC[:] = CC[:]
    
    update(alpha, max_iter, shm_Tstar.name, Tstar.shape, Tstar.dtype,
           shm_CC.name, CC.shape, CC.dtype, barrier)
    
    Tstar[:] = np_Tstar[:]
    
    del np_Tstar
    shm_Tstar.close()
    shm_Tstar.unlink()
    del np_CC
    shm_CC.close()
    shm_CC.unlink()
    
    Zstar = Tstar.dot(Lpinv_BT.transpose())
    
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, Zstar[:,:n].T


def gpm_alignment(y, d, Utilde, C, intermed_param, global_opts):
    CC, Lpinv_BT, D, _ = build_ortho_optim(d, Utilde, intermed_param,
                                            far_off_points=global_opts['far_off_points'],
                                            repel_by=global_opts['repel_by'],
                                            beta=global_opts['beta'])
    CC = D - CC
    M,n = Utilde.shape
    n_proc = min(M,global_opts['n_proc'])
    barrier = mp.Barrier(n_proc)

    def update(alpha, max_iter, shm_name_O, O_shape, O_dtype,
               shm_name_CC, CC_shape, CC_dtype, barrier):
        ###########################################
        # Parallel Updates
        ###########################################
        def target_proc(p_num, chunk_sz, barrier):
            existing_shm_O = shared_memory.SharedMemory(name=shm_name_O)
            O = np.ndarray(O_shape, dtype=O_dtype, buffer=existing_shm_O.buf)
            existing_shm_CC = shared_memory.SharedMemory(name=shm_name_CC)
            CC = np.ndarray(CC_shape, dtype=CC_dtype, buffer=existing_shm_CC.buf)
            
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = M
            else:
                end_ind = (p_num+1)*chunk_sz
            for _ in range(max_iter):
                O_copy = O.copy()
                barrier.wait()
                for i in range(start_ind, end_ind):
                    temp = np.dot(O_copy, CC[:,i*d:(i+1)*d])
                    U_,S_,VT_ = scipy.linalg.svd(temp)
                    O[:,i*d:(i+1)*d] = np.matmul(U_,VT_)
                barrier.wait()
            
            existing_shm_O.close()
            existing_shm_CC.close()
        
        
        proc = []
        chunk_sz = int(M/n_proc)
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz, barrier),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            proc[p_num].join()
        ###########################################
        
        # Sequential version of above
        # for i in range(M):
        #     temp0 = O[:,i*d:(i+1)*d]
        #     temp1 = skew(np.matmul(xi[:,i*d:(i+1)*d],temp0.T))
        #     Q_,R_ = unique_qr(temp0 - t*np.matmul(temp1,temp0))
        #     O[:,i*d:(i+1)*d] = Q_

    alpha = global_opts['alpha']
    max_iter = global_opts['max_internal_iter']
    Tstar = np.zeros((d,M*d))
    for s in range(M):
        Tstar[:,s*d:(s+1)*d] = np.eye(d)
    
    print('Descent starts', flush=True)
    shm_Tstar = shared_memory.SharedMemory(create=True, size=Tstar.nbytes)
    np_Tstar = np.ndarray(Tstar.shape, dtype=Tstar.dtype, buffer=shm_Tstar.buf)
    np_Tstar[:] = Tstar[:]
    shm_CC = shared_memory.SharedMemory(create=True, size=CC.nbytes)
    np_CC = np.ndarray(CC.shape, dtype=CC.dtype, buffer=shm_CC.buf)
    np_CC[:] = CC[:]
    
    update(alpha, max_iter, shm_Tstar.name, Tstar.shape, Tstar.dtype,
           shm_CC.name, CC.shape, CC.dtype, barrier)
    
    Tstar[:] = np_Tstar[:]
    
    del np_Tstar
    shm_Tstar.close()
    shm_Tstar.unlink()
    del np_CC
    shm_CC.close()
    shm_CC.unlink()
    
    Zstar = Tstar.dot(Lpinv_BT.transpose())
    
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})

    return y, Zstar[:,:n].T

def vec(A):
    A = A.copy()
    N = A.shape[0]
    A = np.multiply(A, np.sqrt(2)*(1-np.eye(N))+np.eye(N))
    return np.array(A[np.triu_indices(N)]).flatten()

def sdp_alignment(y, d, Utilde,
                  C, intermed_param, global_opts,
                  seq_of_intermed_views_in_cluster,
                  solver=None):
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=global_opts['far_off_points'],
                                     repel_by=global_opts['repel_by'],
                                     beta=global_opts['beta'])
    if (global_opts['n_repel'] > 0) and global_opts['repel_by'] > 0:
        di = np.diag_indices(4)
        CC[di] += np.linalg.norm(CC)
    M,n = Utilde.shape
    b = vec(CC)
    if solver is None:
        c_mat = (1-np.tri(M*d,k=-1))*np.tri(M*d,k=1)
        c_mat = c_mat[np.triu_indices(M*d)]
        inds = np.where(c_mat)[0]
        n_ = M*d+M*d-1
        A = csc_matrix((np.ones(n_), (inds, np.arange(n_))),
                        shape=((M*d*(M*d+1))//2,n_))
        c = np.arange(n_)
        c = -1.0*(c%2==0)
        
        data = {'P':None,
                'b': b,
                'A': A,
                'c': c}

        cone = dict(s=[M*d])
        solver = scs.SCS(data, cone, mkl=True, eps_abs=global_opts['eps'],
                        max_iters=global_opts['max_internal_iter'])
    else:
        print('Reusing solver by updating b')
        solver.update(b=b)
    print('SDP solver starts', flush=True)
    sol = solver.solve(warm_start=True, y=vec(np.tile(np.eye(d), (M,M))))
    
    Y = np.zeros((M*d,M*d))
    Y[np.triu_indices(M*d)] = sol['y']
    Y_diag = np.diag(Y)
    Y = Y + Y.T
    Y = Y/np.sqrt(2)
    np.fill_diagonal(Y, Y_diag)
    #print('Objective value verification', np.trace(np.dot(OTO, Y.T)))
    #B = ichol(Y.copy())
    print(sol['info'], flush=True)
    
    np.random.seed(42)
    v0 = np.random.uniform(0,1,(Y.shape[0]))
    lmbda, B = slinalg.eigsh(Y, k=d, which='LM')
    B = np.sqrt(lmbda)[None,:]*B
    #print('|Y-BB^T|',np.sum(np.abs(Y-np.dot(B,B.T))))
    
    B = B.T
    Tstar = np.zeros((d,M*d))
    
    n_clusters = len(seq_of_intermed_views_in_cluster)
    for i in range(n_clusters):
        # the first view in each cluster is not rotated
        seq = seq_of_intermed_views_in_cluster[i]
        s0 = seq[0]
        U_,S_,VT_ = scipy.linalg.svd(B[:,d*s0:d*(s0+1)])
        Q =  np.matmul(U_,VT_)
        Q = Q.T
        
        for m in range(M):
            U_,S_,VT_ = scipy.linalg.svd(B[:,d*m:d*(m+1)])
            temp_ = np.matmul(U_,VT_)
            Tstar[:,m*d:(m+1)*d] = np.matmul(temp_, Q)
    
    Zstar = Tstar.dot(Lpinv_BT.transpose())
    
    # the first view in each cluster is not translated
    for i in range(n_clusters):
        seq = seq_of_intermed_views_in_cluster[i]
        s0 = seq[0]
        temp = Zstar[:,n+s0][:,None].copy()
        Zstar[:,n+seq] -= temp
        C_seq = C[seq,:].sum(axis=0).nonzero()[1]
        Zstar[:,C_seq] -= temp
    
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    
    return y, Zstar[:,:n].T, solver