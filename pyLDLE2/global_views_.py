import pdb
import time
import numpy as np
import copy

from .util_ import procrustes, print_log
from .global_reg_ import sequential_init, spectral_init, sequential_final, retraction_final, compute_alignment_err

from scipy.linalg import svdvals
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

class GlobalViews:
    def __init__(self, exit_at, print_logs=True, debug=False):
        self.exit_at = exit_at
        self.print_logs = print_logs
        self.debug = debug
        
        self.y_init = None
        self.color_of_pts_on_tear_init = None
        self.y_final = None
        self.color_of_pts_on_tear_final = None
        self.tracker = {}
        
        self.local_start_time = time.time()
        self.global_start_time = time.time()
        
        # saved only when debug is True
        self.first_intermed_view_in_cluster = None
        self.n_Utilde_Utilde = None
        self.seq_of_intermed_views_in_cluster = None
        self.parents_of_intermed_views_in_cluster = None
        self.y_seq_init = None
        self.y_spec_init = None
        self.contrib_of_view = None
        self.y_refined_at = []
        self.color_of_pts_on_tear_at = []
        
    def log(self, s='', log_time=False):
        if self.print_logs:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
            
    def fit(self, d, Utilde, C, c, n_C, intermed_param, global_opts, vis, vis_opts):
        if global_opts['main_algo'] == 'LDLE':
            # Compute |Utilde_{mm'}|
            n_Utilde_Utilde = np.dot(Utilde, Utilde.T)
            np.fill_diagonal(n_Utilde_Utilde, 0)
            
            # Compute sequence of intermedieate views
            seq_of_intermed_views_in_cluster, \
            parents_of_intermed_views_in_cluster, \
            cluster_of_intermed_view = self.compute_seq_of_intermediate_views(Utilde, n_C, 
                                                                             n_Utilde_Utilde,
                                                                             intermed_param)
            
            # Visualize embedding before init
            if global_opts['vis_before_init']:
                self.vis_embedding_(d, intermed_param, C, Utilde,
                                  n_Utilde_Utilde, global_opts, vis,
                                  vis_opts, title='Before_Init')
            
            # Compute initial embedding
            y_init, first_intermed_view_in_cluster,\
            color_of_pts_on_tear_init = self.compute_init_embedding(d, Utilde, n_Utilde_Utilde, intermed_param,
                                                                    seq_of_intermed_views_in_cluster,
                                                                    parents_of_intermed_views_in_cluster,
                                                                    C, c, vis, vis_opts, global_opts)

            self.y_init = y_init
            self.color_of_pts_on_tear_init = color_of_pts_on_tear_init
            
            if global_opts['refine_algo_name']:
                y_final,\
                color_of_pts_on_tear_final = self.compute_final_embedding(y_init, d, Utilde, C, intermed_param,
                                                                          n_Utilde_Utilde,
                                                                          first_intermed_view_in_cluster, 
                                                                          seq_of_intermed_views_in_cluster,
                                                                          parents_of_intermed_views_in_cluster, 
                                                                          cluster_of_intermed_view, global_opts,
                                                                          vis, vis_opts)
                self.y_final = y_final
                self.color_of_pts_on_tear_final = color_of_pts_on_tear_final
            
        elif global_opts['main_algo'] == 'LTSA':
            print('Using LTSA Global Alignment...')
#             self.y_final = self.compute_final_global_embedding_ltsap_based()
#             self.color_of_pts_on_tear_final = None
        
        
        
        if self.debug:
            self.first_intermed_view_in_cluster = first_intermed_view_in_cluster
            self.n_Utilde_Utilde = n_Utilde_Utilde
            self.seq_of_intermed_views_in_cluster = seq_of_intermed_views_in_cluster
            self.parents_of_intermed_views_in_cluster = parents_of_intermed_views_in_cluster
    
    def compute_seq_of_intermediate_views(self, Utilde, n_C, n_Utilde_Utilde,
                                          intermed_param, print_prop = 0.25):
        M = Utilde.shape[0]
        print_freq = int(print_prop * M)
        # First intermediate view in the sequence
        s_1 = np.argmax(n_C)

        # W_{mm'} = W_{m'm} measures the ambiguity between
        # the pair of the embeddings of the overlap 
        # Utilde_{mm'} in mth and m'th intermediate views
        W = np.zeros((M,M))

        # Iterate over pairs of overlapping intermediate views
        for m in range(M):
            if np.mod(m, print_freq)==0:
                print('Ambiguous overlaps checked for %d intermediate views' % m)
            for mp in np.where(n_Utilde_Utilde[m,:] > 0)[0].tolist():
                if mp > m:
                    # Compute Utilde_{mm'}
                    Utilde_mmp = Utilde[m,:]*Utilde[mp,:]
                    # Compute V_{mm'}, V_{m'm}, Vbar_{mm'}, Vbar_{m'm}
                    V_mmp = intermed_param.eval_({'view_index': m, 'data_mask': Utilde_mmp})
                    V_mpm = intermed_param.eval_({'view_index': mp, 'data_mask': Utilde_mmp})
                    Vbar_mmp = V_mmp - np.mean(V_mmp,0)[np.newaxis,:]
                    Vbar_mpm = V_mpm - np.mean(V_mpm,0)[np.newaxis,:]
                    # Compute ambiguity as the minimum singular value of
                    # the d x d matrix Vbar_{mm'}^TVbar_{m'm}
                    W[m,mp] = svdvals(np.dot(Vbar_mmp.T,Vbar_mpm))[-1]
                    W[mp,m] = W[m,mp]

        print('Ambiguous overlaps checked for %d points' % M)
        # Compute maximum spanning tree/forest of W
        T = minimum_spanning_tree(coo_matrix(-W))
        # Detect clusters of manifolds and create
        # a sequence of intermediate views for each of them
        n_visited = 0
        seq_of_intermed_views_in_cluster = []
        parents_of_intermed_views_in_cluster = []
        # stores cluster number for the intermediate views in a cluster
        cluster_of_intermed_view = np.zeros(M,dtype=int)
        is_visited = np.zeros(M, dtype=bool)
        cluster_num = 0
        while n_visited < M:
            # First intermediate view in the sequence
            s_1 = np.argmax(n_C * (1-is_visited))
            # Compute breadth first order in T starting from s_1
            s_, rho_ = breadth_first_order(T, s_1, directed=False) #(ignores edge weights)
            seq_of_intermed_views_in_cluster.append(s_)
            parents_of_intermed_views_in_cluster.append(rho_)
            is_visited[s_] = True
            cluster_of_intermed_view[s_] = cluster_num
            n_visited = np.sum(is_visited)
            cluster_num = cluster_num + 1
            
        print('Seq of intermediate views and their predecessors computed.')
        print('No. of connected components =', len(seq_of_intermed_views_in_cluster))
        if len(seq_of_intermed_views_in_cluster)>1:
            print('Multiple connected components detected')
        return seq_of_intermed_views_in_cluster,\
               parents_of_intermed_views_in_cluster,\
               cluster_of_intermed_view
    
    def compute_color_of_pts_on_tear(self, y, Utilde, C, global_opts,
                                     n_Utilde_Utilde, n_Utildeg_Utildeg=None):
        M,n = Utilde.shape

        # Compute |Utildeg_{mm'}| if not provided
        if n_Utildeg_Utildeg is None:
            _, n_Utildeg_Utildeg = self.compute_Utildeg(y, Utilde, C, global_opts)

        color_of_pts_on_tear = np.zeros(n)+np.nan

        # Compute the tear: a graph between views where ith view
        # is connected to jth view if they are neighbors in the
        # ambient space but not in the embedding space
        tear = (n_Utilde_Utilde > 0) & (n_Utildeg_Utildeg == 0)

        # Keep track of visited views across clusters of manifolds
        is_visited = np.zeros(M, dtype=bool)
        n_visited = 0
        while n_visited < M: # boundary of a cluster remain to be colored
            # track the next color to assign
            cur_color = 1

            s0 = np.argmax(is_visited == 0)
            seq, rho = breadth_first_order(n_Utilde_Utilde>0, s0, directed=False) #(ignores edge weights)
            is_visited[seq] = True
            n_visited = np.sum(is_visited)

            # Iterate over views
            for m in seq:
                to_tear_mth_view_with = np.where(tear[m,:])[0].tolist()
                if len(to_tear_mth_view_with):
                    # Points in the overlap of mth view and the views
                    # on the opposite side of the tear
                    temp = Utilde[m,:][np.newaxis,:] & Utilde[to_tear_mth_view_with,:]
                    for i in range(len(to_tear_mth_view_with)):
                        mp = to_tear_mth_view_with[i]
                        # Compute points on the overlap of m and m'th view
                        # which are in mth cluster and in m'th cluster. If
                        # both sets are non-empty then assign them same color.
                        temp_m = temp[i,:] & (C[m,:]) & np.isnan(color_of_pts_on_tear)
                        temp_mp = temp[i,:] & (C[mp,:])  & np.isnan(color_of_pts_on_tear)
                        if np.any(temp_m) and np.any(temp_mp):
                            color_of_pts_on_tear[temp_m|temp_mp] = cur_color
                            cur_color += 1
                        
        return color_of_pts_on_tear
    
    def vis_embedding_(self, d, intermed_param, C, Utilde,
                      n_Utilde_Utilde, global_opts, vis,
                      vis_opts, title='', color_of_pts_on_tear=None):
        M,n = Utilde.shape
        y = np.zeros((n,d))
        for s in range(M):
            C_s = C[s,:]
            y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})

        if (color_of_pts_on_tear is None) and global_opts['to_tear']:
            color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y, Utilde, C, global_opts,
                                                                     n_Utilde_Utilde)
            
        vis.global_embedding(y, vis_opts['c'], vis_opts['cmap_interior'],
                              color_of_pts_on_tear, vis_opts['cmap_boundary'],
                              title)
        plt.show()
        return color_of_pts_on_tear
    
    def vis_embedding(self, y, vis, vis_opts, color_of_pts_on_tear=None, title=''):
        vis.global_embedding(y, vis_opts['c'], vis_opts['cmap_interior'],
                              color_of_pts_on_tear, vis_opts['cmap_boundary'],
                              title)
        plt.show()
        
    
    def compute_init_embedding(self, d, Utilde, n_Utilde_Utilde, intermed_param,
                               seq_of_intermed_views_in_cluster,
                               parents_of_intermed_views_in_cluster,
                               C, c, vis, vis_opts, global_opts,
                               print_prop = 0.25):
        M,n = Utilde.shape
        print_freq = int(M*print_prop)

        intermed_param.T = np.tile(np.eye(d),[M,1,1])
        intermed_param.v = np.zeros((M,d))
        y = np.zeros((n,d))

        n_clusters = len(seq_of_intermed_views_in_cluster)

        # Boolean array to keep track of already visited views
        is_visited_view = np.zeros(M, dtype=bool)
        
        # First intermediate view in each cluster is fixed
        first_intermed_view_in_cluster = []
        for i in range(n_clusters):
            seq = seq_of_intermed_views_in_cluster[i]
            first_intermed_view_in_cluster.append(seq[0])
        
        # sequential init overrides this if global_opts['to_tear'] is True
        contrib_of_view = Utilde.copy()

        init_algo = global_opts['init_algo_name']
        self.log('Computing initial embedding using: ' + init_algo + ' algorithm', log_time=True)
        if 'sequential' in init_algo:
            # Initialization contribution of view i
            # as the points in cluster i
            contrib_of_view = C.copy()
            for i in range(n_clusters):
                # First view global embedding is same as intermediate embedding
                seq_0 = first_intermed_view_in_cluster[i]
                is_visited_view[seq_0] = True
                y[C[seq_0,:],:] = intermed_param.eval_({'view_index': seq_0, 'data_mask': C[seq_0,:]})
                
                seq = seq_of_intermed_views_in_cluster[i]
                rho = parents_of_intermed_views_in_cluster[i]
                y, is_visited_view,\
                contrib_of_view_ = sequential_init(seq, rho, y, is_visited_view,
                                            d, Utilde, n_Utilde_Utilde,
                                            C, c, intermed_param,
                                            global_opts, print_freq,
                                            ret_contrib_of_views=True)
                if global_opts['to_tear']:
                    contrib_of_view[seq,:] = contrib_of_view_[seq,:]
            
            if self.debug:
                self.y_seq_init = y
                self.contrib_of_view = contrib_of_view
        
        if 'spectral' in init_algo:
            y, y_2,\
            is_visited_view = spectral_init(y, is_visited_view,
                                            d, contrib_of_view,
                                            C, intermed_param,
                                            global_opts, print_freq)
            
            if self.debug:
                self.y_spec_init = y
                self.y_spec_init_2 = y_2
                
        
        self.log('Embedding initialized.', log_time=True)
        self.tracker['init_computed_at'] = time.time()
        if global_opts['compute_error']:
            self.log('Computing error.')
            err, _ = compute_alignment_err(d, Utilde, intermed_param)
            self.log('Alignment error: %0.3f' % err, log_time=True)
            self.tracker['init_err'] = err
        
        # arrange connected components nicely
        # spaced on horizontal (x) axis
        offset = 0
        for i in range(n_clusters):
            seq = seq_of_intermed_views_in_cluster[i]
            pts_in_cluster_i = np.any(C[seq,:], axis=0)
            
            # make the x coordinate of the leftmost point
            # of the ith cluster to be equal to the offset
            if i > 0:
                offset_ = np.min(y[pts_in_cluster_i,0])
                intermed_param.v[seq,0] += offset - offset_
            
            # recompute the embeddings of the points in this cluster
            for s in range(seq.shape[0]):
                C_s = C[seq[s],:]
                y[C_s,:] = intermed_param.eval_({'view_index': seq[s], 'data_mask': C_s})
            
            # recompute the offset as the x coordinate of
            # rightmost point of the current cluster
            offset = np.max(y[pts_in_cluster_i,0])
        
        # Visualize the initial embedding
        color_of_pts_on_tear = self.vis_embedding_(d, intermed_param, C, Utilde,
                                                  n_Utilde_Utilde, global_opts, vis,
                                                  vis_opts, title='Init')

        return y, first_intermed_view_in_cluster, color_of_pts_on_tear

    def compute_Utildeg(self, y, Utilde, C, global_opts):
        M,n = Utilde.shape
        d_e_ = squareform(pdist(y)) # O(n^2 d)

        k_ = min(global_opts['k']*global_opts['nu'], d_e_.shape[0]-1)
        
        neigh = NearestNeighbors(n_neighbors=k_,
                                 metric='precomputed',
                                 algorithm='brute')
        neigh.fit(d_e_)
        neigh_dist, _ = neigh.kneighbors()

        epsilon = neigh_dist[:,[k_-1]]
        Ug = d_e_ < (epsilon + 1e-12)
        Utildeg = np.zeros((M,n),dtype=bool)
        # O(n^2)
        for m in range(M):
            Utildeg[m,:] = np.any(Ug[C[m,:],:], 0)
            
        # |Utildeg_{mm'}|
        n_Utildeg_Utildeg = np.dot(Utildeg, Utildeg.T) # O(M^2 n) = O(n^3/eta_min^2)
        np.fill_diagonal(n_Utildeg_Utildeg, 0)
        return Utildeg, n_Utildeg_Utildeg

    def compute_final_embedding(self, y, d, Utilde, C, intermed_param, n_Utilde_Utilde,
                                first_intermed_view_in_cluster, 
                                seq_of_intermed_views_in_cluster,
                                parents_of_intermed_views_in_cluster, 
                                cluster_of_intermed_view, global_opts,
                                vis, vis_opts):
        M,n = Utilde.shape
        y = y.copy()
        n_clusters = len(seq_of_intermed_views_in_cluster)
        # Boolean array to keep track of already visited views
        is_visited_view = np.zeros(M, dtype=bool)
        print_freq = int(M*0.25)

        np.random.seed(42) # for reproducbility

        old_time = time.time()
        # If to tear the closed manifolds
        if global_opts['to_tear']:
            # Compute |Utildeg_{mm'}|
            _, n_Utildeg_Utildeg = self.compute_Utildeg(y, Utilde, C, global_opts)
        else:
            n_Utildeg_Utildeg = None
        
        
        CC = None
        Lpinv = None
        B = None

        max_iter0 = global_opts['max_iter']
        max_iter1 = global_opts['refine_algo_max_internal_iter']
        refine_algo = global_opts['refine_algo_name']
        
        self.tracker['refine_iter_start_at'] = np.zeros(max_iter0)
        self.tracker['refine_iter_done_at'] = np.zeros(max_iter0)
        self.tracker['refine_err_at_iter'] = np.zeros(max_iter0)
        
        contrib_of_view = Utilde.copy()
        # Refine global embedding y
        for it0 in range(max_iter0):
            self.tracker['refine_iter_start_at'][it0] = time.time()
            self.log('Refining with ' + refine_algo + ' algorithm for ' + str(max_iter1) + ' iterations.')
            self.log('Refinement iteration: %d' % it0, log_time=True)
            
            if refine_algo == 'sequential':
                y = sequential_final(y, d, Utilde, C, intermed_param, n_Utilde_Utilde, n_Utildeg_Utildeg,
                                     first_intermed_view_in_cluster,
                                     parents_of_intermed_views_in_cluster, cluster_of_intermed_view,
                                     global_opts)
                    
            elif (refine_algo == 'retraction') or (refine_algo == 'spectral'):
                
                if global_opts['to_tear']:
                    contrib_of_view = C.copy()
                    ZZ = (n_Utilde_Utilde > 0) & (n_Utildeg_Utildeg > 0)
                    for i in range(n_clusters):
                        seq = seq_of_intermed_views_in_cluster[i]
                        rho = parents_of_intermed_views_in_cluster[i]
                        for m in range(1,seq.shape[0]):
                            s = seq[m]
                            Z_s = ZZ[s,:]
                            Z_s = np.where(Z_s)[0].tolist()
                            if len(Z_s) == 0: # ideally this should not happen
                                Z_s = parents_of_intermed_views_in_cluster[cluster_of_intermed_view[s]][s]
                                Z_s = [Z_s]
                            
                            contrib_of_view[s,:] |= (Utilde[s,:] & np.any(C[Z_s,:], 0))
                            
                if refine_algo == 'retraction':
                    y, CCLpinvB = retraction_final(y, d, contrib_of_view, C, intermed_param,
                                                   n_Utilde_Utilde, n_Utildeg_Utildeg,
                                                   first_intermed_view_in_cluster,
                                                   parents_of_intermed_views_in_cluster,
                                                   cluster_of_intermed_view,
                                                   global_opts, CC, Lpinv, B)
                elif refine_algo == 'spectral':
                    y, y_2,\
                    is_visited_view = spectral_init(y, is_visited_view,
                                                    d, contrib_of_view,
                                                    C, intermed_param,
                                                    global_opts, print_freq)
#                 if not global_opts['to_tear']:
#                     CC, Lpinv, B = CCLpinvB
            
            
                
            self.log('Done.', log_time=True)
            self.tracker['refine_iter_done_at'][it0] = time.time()

            if global_opts['compute_error'] or (it0 == max_iter0-1):
                self.log('Computing error.')
                err, _ = compute_alignment_err(d, Utilde, intermed_param)
                self.log('Alignment error: %0.3f' % err, log_time=True)
                self.tracker['refine_err_at_iter'][it0] = err
            
            # If to tear the closed manifolds
            if global_opts['to_tear']:
                # Compute |Utildeg_{mm'}|
                _, n_Utildeg_Utildeg = self.compute_Utildeg(y, Utilde, C, global_opts)
                color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y, Utilde, C, global_opts,
                                                                         n_Utilde_Utilde,
                                                                         n_Utildeg_Utildeg)
            else:
                color_of_pts_on_tear = None
                
            if self.debug:
                self.y_refined_at.append(y)
                self.color_of_pts_on_tear_at.append(color_of_pts_on_tear)
                
             
            # Visualize the current embedding
            self.vis_embedding_(d, intermed_param, C, Utilde,
                              n_Utilde_Utilde, global_opts, vis,
                              vis_opts, title='Iter_%d' % it0,
                              color_of_pts_on_tear=color_of_pts_on_tear)

        return y, color_of_pts_on_tear