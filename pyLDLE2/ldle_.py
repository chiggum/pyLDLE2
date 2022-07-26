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
import multiprocess as mp
from pandas import DataFrame
    
def double_manifold_k_nn(data, ddX, k_nn, metric, n_proc=1):
    """Doubles the manifold represented by X and computes
    k-nearest neighbors of each point on the double.
    
    Parameters
    ----------
    data : {array} of shape (n_samples, n_features) or {array} of shape (n_samples, n_samples)
           A 2d array where each row represents a data point or
           distance of a point to all other points.
    ddX : array shape (n_samples,)
          A 1d np.array of shape (n,) where ddX[i] = 0 iff the
          i-th point X[i,:] represents a point on the boundary
          of the manifold. Note that n_boundary = np.sum(ddX==0).
    k_nn : int
           Number of nearest neighbors to compute on the double.
    n_proc : int
             Number of processors to use.
    Returns
    -------
    neigh_dist : array shape (2*n_samples-n_boundary, k_nn)
                 distance of k-nearest neighbors from each
                 point on the double.
    neigh_ind : array shape (2*n_samples-n_boundary, k_nn)
                indices of k-nearest neighbors from each point
                on the double.
    """
    n = data.shape[0] # no. of points on the original manifold
    dX = ddX==0 # a boolean array s.t. dX[i] == 1 iff i-th pt is on boundary
    n_dX = np.sum(dX) # no. of points on the boundary
    print('No. of points on the boundary =', n_dX)
    n_interior = n-n_dX # no. of points in the interior

    # For points in the interior, a duplicate point
    # with index offset-ed by n is created.
    # The i-th point in the interior of the original
    # manifold is duplicated. The index of the duplicate
    # is n + i.
    ind_mask = np.arange(n)
    ind_mask[~dX] = n+np.arange(n_interior)
    
    k_nn_ = 2*k_nn
    neigh_dist, neigh_ind = nearest_neighbors(data, k_nn_, metric)
    
    # Now we construct a sparse distance matrix corresp
    # to the doubled manifold. The block structure is
    # A11 A12 A13
    # A21 A22 A23
    # A31 A32 A33
    # 1: interior of original manifold
    # 2: boundary
    # 3: duplicated interior
    # So, A11 = distance matrix between pairs of
    # points in the interior of original manifold.
    # Overall, A11, A12, A21, A22, A32, A33 
    # are known or trivially computable. A13 = A31.T
    # is to be computed.
    
    # A22, A23
    # A32 A33
    neigh_ind_ = ind_mask[neigh_ind]
    neigh_dist_ = neigh_dist
    
    dX_of_neigh_ind = dX[neigh_ind]
    pts_near_dX = np.any(dX_of_neigh_ind, 1)
    int_pts_near_dX = pts_near_dX & (~dX)
    int_pts_near_dX_inds = np.where(int_pts_near_dX)[0]
    
    # Compute sparse representation of A13
    row_inds13 = []
    col_inds13 = []
    vals13 = []
    
    def target_proc(p_num, chunk_sz, q_):
        start_ind = p_num*chunk_sz
        if p_num == (n_proc-1):
            end_ind = int_pts_near_dX_inds.shape[0]
        else:
            end_ind = (p_num+1)*chunk_sz

        row_inds13_ = []
        col_inds13_ = []
        vals13_ = []
        for i in range(start_ind, end_ind):
            # index of an interior point which has
            # a boundary point as nbr
            k = int_pts_near_dX_inds[i]

            # nbrs of k which are on the boundary
            # and the distance of k to these points.
            nbrs_on_dX_mask = dX_of_neigh_ind[k,:]
            nbrs_on_dX = neigh_ind[k,nbrs_on_dX_mask]
            dist_of_nbrs_on_dX = neigh_dist[k,nbrs_on_dX_mask][:,None]

            # Let S_k be the the nbrs of k which are
            # on the boundary. The nbrs of S_k in the
            # duplicated manifold are
            nbrs_of_S_k = neigh_ind_[nbrs_on_dX,:]
            # Distance of nbrs of S_k to the corresponding
            # point in S_k.
            dist_of_nbrs_of_S_k = neigh_dist_[nbrs_on_dX,:]

            # Distance b/w k and the nbrs of S_k
            dist_of_k_from_nbrs_of_S_k = dist_of_nbrs_of_S_k + dist_of_nbrs_on_dX

            # nbrs of S_k on the duplicated mainfold
            # which are in the interior of it and
            # distanace of k to these points.
            int_nbrs_of_S_k_mask = ~dX_of_neigh_ind[nbrs_on_dX,:]
            int_nbrs_of_S_k = nbrs_of_S_k[int_nbrs_of_S_k_mask]
            dist_of_k_from_int_nbrs_of_S_k = dist_of_k_from_nbrs_of_S_k[int_nbrs_of_S_k_mask]

            # Groupby interior point ids and min-aggregate their distances
            df = DataFrame(dict(ind=int_nbrs_of_S_k,
                               dist=dist_of_k_from_int_nbrs_of_S_k))
            gp = df.groupby('ind')['dist'].min().to_frame().reset_index()
            col_inds_ = gp['ind'].to_list()
            vals_ = gp['dist'].to_list()
            row_inds_ = [k]*len(col_inds_)
            
            row_inds13_ += row_inds_
            col_inds13_ += col_inds_
            vals13_ += vals_

        q_.put((row_inds13_, col_inds13_, vals13_))

    q_ = mp.Queue()
    chunk_sz = int(int_pts_near_dX_inds.shape[0]/n_proc)
    proc = []
    for p_num in range(n_proc):
        proc.append(mp.Process(target=target_proc,
                               args=(p_num,chunk_sz,q_),
                               daemon=True))
        proc[-1].start()

    for p_num in range(n_proc):
        row_inds_, col_inds_, vals_ = q_.get()
        row_inds13 += row_inds_
        col_inds13 += col_inds_
        vals13 += vals_

    q_.close()

    for p_num in range(n_proc):
        proc[p_num].join()
    
    row_inds22_23_32_33 = np.repeat(ind_mask, neigh_ind.shape[1]).tolist()
    col_inds22_23_32_33 = ind_mask[neigh_ind].flatten().tolist()
    vals_22_23_32_33 = neigh_dist.flatten().tolist()
    
    row_inds11_12 = np.repeat(np.arange(n)[~dX], neigh_ind.shape[1]).tolist()
    col_inds11_12 = neigh_ind[~dX,:].flatten().tolist()
    vals11_12 = neigh_dist[~dX,:].flatten().tolist()
    
    cs_graph = csr_matrix((vals11_12+vals13+vals_22_23_32_33,
                           (row_inds11_12+row_inds13+row_inds22_23_32_33,
                            col_inds11_12+col_inds13+col_inds22_23_32_33)),
                          shape=(n_interior+n, n_interior+n))
    cs_graph = cs_graph.maximum(cs_graph.transpose())
    
    neigh_dist, neigh_ind = nearest_neighbors(cs_graph, k_nn=k_nn, metric='precomputed')
    return neigh_dist, neigh_ind
    

def get_default_local_opts(algo='LDLE', k_nn=49, k_tune=7, k=24, gl_type='unnorm',
                           N=100, no_gamma=False, Atilde_method='LDLE_1',
                           p=0.99, tau=50, delta=0.9, to_postprocess= True,
                           pp_n_thresh=32):
    """Sets and returns a dictionary of default_local_opts.
    
    Parameters
    ----------
    algo : str
           The algorithm to use for the construction of
           local views. Options are 'LDLE' and 'LTSA'.
           LTSA uses the hyperparameter k only and is
           not affected by the value of the others.
    k_nn : int
           For k-nearest neighbor graph construction.
    k_tune : int
           Distance to k_tune-th nearest neighbor is
           used as the local scaling factor. This must
           be less than k_nn.
    k : int 
        The size of local view per point.
    gl_type : str
              The type of graph Laplacian to construct.
              Options are 'unnorm' for unnormalized,
              'symnorm' for symmetric normalized,
              'diffusion' for density-corrected normalized.
    N : int
        Number of smallest non-trivial eigenvectors of the
        graph Laplacian to be used for the construction of
        local views.
    no_gamma : bool
               If True, the parameterization of local views
               are not normalized.
    Atilde_method : str
                    Method to use for conmputing Atilde.
                    Options are 'LDLE_1' for finite element method,
                    'LLR' for local linear regression. Currently,
                    only 'LDLE_1' works.
    p : float
        A hyperparameter used in computing Atilde. The value
        must be in (0,1).
    tau : int
        A hyperparameter used in computing parameterizations
        of local views. The value must be in (0,100).
    delta : float
        A hyperparameter used in computing parameterizations
        of local views. The value must be in (0,1).
    to_postprocess : bool
        If True the local parameterizations are postprocessed
        to fix anamolous parameterizations leading to high
        distortion of the local views.
    pp_n_thresh : int
        Threshold to use multiple processors or a single processor
        while postprocessing the local parameterizations. A small
        value such as 32 leads to faster postprocessing.
    """
    return {'k_nn': k_nn, 'k_tune': k_tune, 'k': k,
           'gl_type': gl_type, 'N': N, 'no_gamma': no_gamma,
           'Atilde_method': Atilde_method, 'p': p, 'tau': tau,
           'delta': delta, 'to_postprocess': to_postprocess, 'algo': algo,
           'pp_n_thresh': pp_n_thresh}

def get_default_intermed_opts(eta_min=5, eta_max=25, len_S_thresh=256):
    """Sets and returns a dictionary of default_intermed_opts.
    
    Parameters
    ----------
    eta_min : int
              Minimum allowed size of the clusters underlying
              the intermediate views. The values must be >= 1.
    eta_max : int
              Maximum allowed size of the clusters underlying
              the intermediate views. The value must be > eta_min.
    len_S_thresh : int
                   Threshold on the number of points for which 
                   the costs are to be updated, to invoke
                   multiple processors.
    """
    return {'eta_min': eta_min, 'eta_max': eta_max, 'len_S_thresh': len_S_thresh}
    
def get_default_global_opts(main_algo='LDLE', to_tear=True, nu=3, max_iter=10, color_tear=True,
                            vis_before_init=False, compute_error=False,
                            init_algo_name='sequential', init_algo_align_w_parent_only=True,
                            refine_algo_name='retraction',
                            refine_algo_max_internal_iter=100,
                            refine_algo_alpha=0.3):
    """Sets and returns a dictionary of default_global_opts.

    Parameters
    ----------
    main_algo : str
                The algorithm to use for the alignment of intermediate
                views. Options are 'LDLE' and 'LTSA'. If 'LTSA' is
                chosen then none of the following hypermateters are used.
    to_tear : bool
              If True the tearing of the manifold is allowed.
    nu : int
         A hyperparameter used to compute neighboring intermediate
         views in the embedding space.
    max_iter : int
               Number of iterations to refine the global embedding for.
    color_tear : bool
                 If True, colors the tear with the colormap provided in vis_opts.
    vis_before_init : bool
                      If True, plots the global embedding before
                      alignment begins. This is same as just plotting
                      all the intermediate views without alignment.
    compute_error : bool
                    If True the alignment error is computed at each 
                    iteration of the refinement, otherwise only at
                    the last iteration.
    init_algo_name : str
                     The algorithm used to compute initial global embedding
                     by aligning the intermediate views.
                     Options are 'sequential' for tree-based-procrustes alignment,
                     'spectral' for spectral alignment (ignores to_tear),
                     'sequential+spectral' for tree-based-procrustes alignment
                     followed by spectral alignment.
    init_algo_align_w_parent_only : bool
                                    If True only the parents of the intermediate
                                    views are used in the tree-based-procrustes
                                    alignment.
    refine_algo_name : str
                       The algorithm used to refine the initial global embedding
                       by refining the alignment between intermediate views.
                       Options are 'sequential' for Generalized Procustes Analysis
                       (GPA) based alignment, 'retraction' for Riemannian optimization
                       based alignment, 'spectral' for spectral alignment.
    refine_algo_max_internal_iter : int
                                    The number of internal iterations used by
                                    the refinement algorithm. This is ignored
                                    by 'spectral' refinement.
    refine_algo_alpha : int
                        The step size used in the Riemannian gradient descent
                        when the refinement algorithm is 'retraction'.
    """
    return {'to_tear': to_tear, 'nu': nu, 'max_iter': max_iter,
               'color_tear': color_tear,
               'vis_before_init': vis_before_init,
               'compute_error': compute_error,
               'main_algo': main_algo, # ['LDLE', 'LTSA']
               'init_algo_name': init_algo_name, # ['sequential', 'spectral', 'sequential+spectral'], spectral ignores to_tear
               'init_algo_align_w_parent_only': init_algo_align_w_parent_only,
               'refine_algo_name': refine_algo_name, # ['sequential', 'retraction', 'spectral']
               'refine_algo_max_internal_iter': refine_algo_max_internal_iter, # 10 for sequential and 100 for retraction
               'refine_algo_alpha': refine_algo_alpha, # step size for retraction
              }
def get_default_vis_opts(save_dir='', cmap_interior='summer', cmap_boundary='jet', c=None):
    """Sets and returns a dictionary of default_vis_opts.
    
    Parameters
    ----------
    save_dir : str
               The directory to save the plots in.
    cmap_interior : str
                    The colormap to use for the interior of the manifold.
    cmap_boundary : str
                    The colormap to use for the boundary of the manifold.
    c : array shape (n_samples)
        The labels for each point to be used to color them.
    """
    return {'save_dir': save_dir,
             'cmap_interior': cmap_interior,
             'cmap_boundary': cmap_boundary,
             'c': c}


class LDLE:
    """Low Dimensional Local Eigenmaps and its variations.
    
    Parameters
    ----------
    d : int
       Intrinsic dimension of the manifold.
    local_opts : dict
                Options for local views construction. The key-value pairs
                provided override the ones in default_local_opts.
    intermed_opts : dict
                    Options for intermediate views construction. The key-value pairs
                    provided override the ones in default_intermed_opts.
    global_opts : dict
                  Options for global views construction. The key-value pairs
                  provided override the ones in default_global_opts.
    vis_opts : dict
               Options for visualization. The key-value pairs
               provided override the ones in default_vis_opts.
    n_proc : int
             The number of processors to use. Defaults to approximately
             3/4th of the available processors. 
    verbose : bool
              print logs if True.
    debug : bool
            saves intermediary objects/data for debugging.
    """
    def __init__(self,
                 d = 2,
                 local_opts = {}, 
                 intermed_opts = {},
                 global_opts = {},
                 vis_opts = {},
                 n_proc = min(8,max(1,int(mp.cpu_count()*0.75))),
                 exit_at = None,
                 verbose = False,
                 debug = False):
        default_local_opts = get_default_local_opts()
        default_intermed_opts = get_default_intermed_opts()
        default_global_opts = get_default_global_opts()
        default_vis_opts = get_default_vis_opts()
        
        self.d = d
        local_opts['n_proc'] = n_proc
        for i in local_opts:
            default_local_opts[i] = local_opts[i]
        self.local_opts = default_local_opts
        #############################################
        intermed_opts['algo'] = self.local_opts['algo']
        intermed_opts['n_proc'] = n_proc
        for i in intermed_opts:
            default_intermed_opts[i] = intermed_opts[i]
        self.intermed_opts = default_intermed_opts
        # Update k_nn 
        self.local_opts['k_nn0'] = max(self.local_opts['k_nn'],
                                       self.intermed_opts['eta_max']*self.local_opts['k'])
        print("local_opts['k_nn0'] =", self.local_opts['k_nn0'], "is created.")
        #############################################
        global_opts['k'] = self.local_opts['k']
        global_opts['n_proc'] = n_proc
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
        """Run the algorithm. Either X or d_e must be supplied.
        
        Parameters
        ---------
        X : array shape (n_samples, n_features)
            A 2d array containing data representing a manifold.
        d_e : array shape (n_samples, n_samples)
            A square numpy matrix representing the geodesic distance
            between each pair of points on the manifold.
        ddX : array shape (n_samples)
            Optional. A 1d array representing the distance of the
            points from the boundary OR a 1d array such that 
            ddX[i] = 0 if the point i i.e. X[i,:] is on the boundary.
            
        Returns
        -------
        y : array shape (n_samples, d)
            The embedding of the data in lower dimension.
        """
        assert X is not None or d_e is not None, "Either X or d_e should be provided."
        
        if d_e is None:
            data = X
        else:
            data = d_e.copy()
        
        if ddX is None or self.local_opts['algo'] == 'LTSA':
            neigh_dist, neigh_ind = nearest_neighbors(data,
                                                      k_nn=self.local_opts['k_nn0'],
                                                      metric='euclidean')
        else:
            self.log('Doubling manifold.')
            neigh_dist, neigh_ind = double_manifold_k_nn(data, ddX,
                                                         self.local_opts['k_nn0'],
                                                         'euclidean',
                                                         self.local_opts['n_proc'])
            self.log('Done.', log_time=True)
        
        # Construct a sparse d_e matrix based on neigh_ind and neigh_dist
        d_e = sparse_matrix(neigh_ind, neigh_dist)
        d_e = d_e.maximum(d_e.transpose())

        neigh_ind = neigh_ind[:,:self.local_opts['k_nn']]
        neigh_dist = neigh_dist[:,:self.local_opts['k_nn']]
        
        # Construct low dimensional local views
        LocalViews = local_views_.LocalViews(self.exit_at, self.verbose, self.debug)
        LocalViews.fit(self.d, X, d_e, neigh_dist, neigh_ind, ddX, self.local_opts)
        
        # Halving distance matrix
        if ddX is not None:
            n = ddX.shape[0]
            d_e = d_e[:n,:n]
        
        self.LocalViews = LocalViews
        if self.exit_at == 'local_views':
            return
        
        # Construct intermediate views
        IntermedViews = intermed_views_.IntermedViews(self.exit_at, self.verbose, self.debug)
        IntermedViews.fit(self.d, d_e, LocalViews.U,
                          LocalViews.local_param_post, self.intermed_opts)
        
        self.IntermedViews = IntermedViews
        if self.exit_at == 'intermed_views':
            return
        
        # Construct Global views
        GlobalViews = global_views_.GlobalViews(self.exit_at, self.verbose, self.debug)
        GlobalViews.fit(self.d, IntermedViews.Utilde, IntermedViews.C, IntermedViews.c,
                        IntermedViews.n_C, IntermedViews.intermed_param, self.global_opts,
                        self.vis, self.vis_opts)
        
        self.GlobalViews = GlobalViews
        
        if self.debug:
            self.d_e = d_e
            self.neigh_ind = neigh_ind
            self.neigh_dist = neigh_dist
        
        return GlobalViews.y_final