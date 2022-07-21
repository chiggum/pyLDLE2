import pdb
import numpy as np
import time

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors, KNeighborsTransformer

from scipy.sparse import csr_matrix

from . import local_views_
from . import intermed_views_
from . import global_views_
from . import visualize_
from .util_ import print_log, sparse_matrix, nearest_neighbors
import multiprocessing
    
def double_manifold(X, ddX, k_nn):
    k_nn_ = 2*k_nn
    neigh = KNeighborsTransformer(n_neighbors=k_nn_, n_jobs=-1) #start with 2 x k_nn
    X_dist_graph = neigh.fit_transform(X)
    
    # Compute mask of points on the boundary
    n = X.shape[0]
    dX = ddX==0
    n_dX = np.sum(dX)
    print('No. of points on the boundary =', n_dX)

    # For points not on the boundary, a duplicate point
    # with index offset-ed by n is created
    ind_mask = np.arange(n)
    ind_mask[~dX] = n+np.arange(n-n_dX)
    
    # eliminate the zeros in the original nearest neighbor graph
    # this basically removes the diagonal entries
    X_dist_graph.eliminate_zeros()
    
    # find row and col indices of non-zero entries
    row_inds = []
    col_inds = []
    for row, col in zip(*X_dist_graph.nonzero()):
        row_inds.append(row)
        col_inds.append(col)
        
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    dist_data = X_dist_graph.data
    
    # a mask of row_inds. True means
    # the row index corresponds to point on the boundary
    dX_inds_mask = dX[row_inds]
    
    # the row indices of points not on the boundary
    # in the duplicated manifold. col indices are computed
    # using ind_mask.
    row_inds2 = ind_mask[row_inds[~dX_inds_mask]]
    col_inds2 = ind_mask[col_inds[~dX_inds_mask]]
    dist_data2 = dist_data[~dX_inds_mask]
    
    # Initialize the doubled graph
    row_inds = np.concatenate([row_inds, row_inds2, np.arange(2*n-n_dX)])
    col_inds = np.concatenate([col_inds, col_inds2, np.arange(2*n-n_dX)])
    dist_data = np.concatenate([dist_data, dist_data2, np.zeros(2*n-n_dX)])
    
    # free up memory
    del row_inds2, col_inds2, dist_data2, X_dist_graph
    
    X_dist_graph2 = csr_matrix((dist_data, (row_inds, col_inds)), shape=(2*n-n_dX, 2*n-n_dX))
    
    # connect points near the boundary on the original
    # and the duplicated manifold
    dX_inds = np.where(dX)[0]
    is_visited = np.zeros((2*n-n_dX,2*n-n_dX), dtype=bool)
    edges = {}
    for i in range(n_dX):
        k = dX_inds[i]
        col_k = X_dist_graph2.getcol(k)
        nbrs = col_k.indices.tolist()
        dists = col_k.data.tolist()
        n_nbrs = len(nbrs)
        for i_ in range(n_nbrs):
            nbr_i = nbrs[i_]
            for j_ in range(i_+1, n_nbrs):
                nbr_j = nbrs[j_]
                if not is_visited[nbr_i,nbr_j]:
                    edges[(nbr_i,nbr_j)] = dists[i_]+dists[j_]
                    edges[(nbr_j,nbr_i)] = edges[(nbr_i,nbr_j)]
                else:
                    edges[(nbr_i,nbr_j)] = np.min([edges[(nbr_i,nbr_j)], dists[i_]+dists[j_]])
                    edges[(nbr_j,nbr_i)] = edges[(nbr_i,nbr_j)]
                    is_visited[nbr_i,nbr_j] = True
                    is_visited[nbr_j,nbr_i] = True
            
    
    del is_visited
    
    row_inds3 = []
    col_inds3 = []
    dist_data3 = []
    for edge, val in edges.items():
        row_inds3.append(edge[0])
        col_inds3.append(edge[1])
        dist_data3.append(val)
    
    # add self connections with approx zero weight
    for i in range(2*n-n_dX):
        row_inds3.append(i)
        col_inds3.append(i)
        dist_data3.append(1e-12)
        
    # Build dinal graph of doubled manifold
    row_inds = np.concatenate([row_inds, np.array(row_inds3)])
    col_inds = np.concatenate([col_inds, np.array(col_inds3)])
    dist_data = np.concatenate([dist_data, np.array(dist_data3)])
    
    # free up memory
    del edges, row_inds3, col_inds3, dist_data3
    
    X_dist_graph3 = csr_matrix((dist_data, (row_inds, col_inds)), shape=(2*n-n_dX, 2*n-n_dX))
    
    pdb.set_trace()

    neigh_dist, neigh_ind = nearest_neighbors(X_dist_graph3, k_nn=k_nn, metric='precomputed')
    return neigh_dist, neigh_ind
    

default_local_opts = {'k_nn': 48, 'k_tune': 6, 'k': 24,
                       'gl_type': 'unnorm', 'N': 100, 'no_gamma': False,
                       'Atilde_method': 'LDLE_1', 'p': 0.99, 'tau': 50,
                       'delta': 0.9, 'to_postprocess': True, 'algo': 'LDLE',
                       'n_proc': max(1,int(multiprocessing.cpu_count()*0.75)),
                       'pp_n_thresh': 32}

default_intermed_opts = {'eta_min': 5, 'eta_max': 100, 'len_S_thresh': 256,}

default_global_opts = {'to_tear': True, 'nu': 3, 'max_iter': 10,
                       'vis_before_init': False,
                       'compute_error': False,
                       'main_algo': 'LDLE', # ['LDLE', 'LTSA']
                       'init_algo': {
                           'name': 'sequential', # ['sequential', 'spectral', 'sequential+spectral'], spectral ignores to_tear
                           'align_w_parent_only': True
                        },
                       'refine_algo': {
                           'name': 'retraction', # ['sequential', 'retraction', 'spectral']
                           'max_internal_iter': 100, # 10 for sequential and 100 for retraction
                           'alpha': 0.3, # step size for retraction
                        }
                      }

default_vis_opts = {'save_dir': '',
                     'cmap_interior': 'summer',
                     'cmap_boundary': 'jet',
                     'c': None}

class LDLE:
    """LDLE 
        :param int d: Embedding dimension. [default=2]
        :param dict local_opts: Options for local views construction. The key-value pairs
                                provided override the ones in default_local_opts.
        :param dict intermed_opts: Options for intermediate views construction. The key-value pairs
                                provided override the ones in default_intermed_opts.
        :param dict global_opts: Options for global views construction. The key-value pairs
                                provided override the ones in default_global_opts.
        :param dict global_opts: Options for global views construction. The key-value pairs
                                provided override the ones in default_global_opts.
        :param bool verbose: print logs if True. [default=False]
        :param bool debug: saves intermediate objects/data for debugging. [default=False]
        :rtype: object
    """
    def __init__(self,
                 d = 2, # embedding dimension
                 local_opts = {}, # see default_local_opts above
                 intermed_opts = {}, # see default_intermed_opts above
                 global_opts = {},# see default_global_opts above
                 vis_opts = {}, # see default_vis_opts above
                 exit_at = None,
                 verbose = False,
                 debug = False):

        self.d = d
        for i in local_opts:
            default_local_opts[i] = local_opts[i]
        self.local_opts = default_local_opts
        self.local_opts['k_nn'] = max(self.local_opts['k_nn'],
                                      self.local_opts['k'])
        #############################################
        intermed_opts['algo'] = self.local_opts['algo']
        intermed_opts['n_proc'] = self.local_opts['n_proc']
        for i in intermed_opts:
            default_intermed_opts[i] = intermed_opts[i]
        self.intermed_opts = default_intermed_opts
        #############################################
        global_opts['k'] = self.local_opts['k']
        global_opts['n_proc'] = self.local_opts['n_proc']
        for i in global_opts:
            default_global_opts[i] = global_opts[i]
        self.global_opts = default_global_opts
        #############################################
        for i in vis_opts:
            default_vis_opts[i] = vis_opts[i]
        self.vis_opts = default_vis_opts
        self.vis = visualize_.Visualize(self.vis_opts['save_dir'])
            
        #############################################
        self.exit_at = exit_at
        self.verbose = verbose
        self.debug = debug
        #############################################
        
        # Other useful inits
        self.global_start_time = time.time()
        self.local_start_time = time.time()
        
        # The variables created during the fit
        self.X = None
        self.d_e = None
        self.ddX = None
        self.neigh_dist = None
        self.neigh_ind = None
        self.LocalViews = None
        self.IntermedViews = None
        
    def log(self, s='', log_time=False):
        if self.verbose:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    def fit(self, X = None, d_e = None, ddX = None):
        assert X is not None or d_e is not None, "Either X or d_e should be provided."
        
        if d_e is None:
            if ddX is None or self.local_opts['algo'] == 'LTSA':
                neigh_dist, neigh_ind = nearest_neighbors(X,
                                                          k_nn=self.local_opts['k_nn'],
                                                          metric='euclidean')
            else:
                self.log('Doubling manifold.')
                neigh_dist, neigh_ind = double_manifold(X, ddX, self.local_opts['k_nn'])
                self.log('Done.', log_time=True)
        
        # Construct a sparse d_e matrix based on neigh_ind and neigh_dist
        d_e = sparse_matrix(neigh_ind, neigh_dist)
        d_e = d_e.maximum(d_e.transpose())
        
        # Construct low dimensional local views
        LocalViews = local_views_.LocalViews(self.exit_at, self.verbose, self.debug)
        LocalViews.fit(self.d, X, d_e, neigh_dist, neigh_ind, ddX, self.local_opts)
        
        # Halving distance matrix
        if ddX is not None:
            n = ddX.shape[0]
            d_e = d_e[:n,:n]
        
        # Construct intermediate views
        IntermedViews = intermed_views_.IntermedViews(self.exit_at, self.verbose, self.debug)
        IntermedViews.fit(self.d, d_e, neigh_ind[:,:self.local_opts['k']], LocalViews.U,
                          LocalViews.local_param_post, self.intermed_opts)
        
        # Construct Global views
        GlobalViews = global_views_.GlobalViews(self.exit_at, self.verbose, self.debug)
        GlobalViews.fit(self.d, IntermedViews.Utilde, IntermedViews.C, IntermedViews.c,
                        IntermedViews.n_C, IntermedViews.intermed_param, self.global_opts,
                        self.vis, self.vis_opts)
        
        self.LocalViews = LocalViews
        self.IntermedViews = IntermedViews
        self.GlobalViews = GlobalViews
        
        if self.debug:
            self.d_e = d_e
            self.neigh_ind = neigh_ind
            self.neigh_dist = neigh_dist