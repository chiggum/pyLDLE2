import pickle
import numpy as np
from . import visualize_

def visualize(fpath, threshs=[5,10,15], n_views=4, figsize1=(8,8),
              figsize2=(16,8), figsize3=(15,10), s1=30, puppets_data=False,
              interact=False):
    with open(fpath, "rb") as f:
        all_data = pickle.load(f)
    X, labelsMat, buml_obj = all_data[:3]

    buml_obj.vis = visualize_.Visualize(buml_obj.vis_opts['save_dir'])

    print('#'*50, flush=True)
    print('Data', flush=True)
    print('#'*50, flush=True)
    if X.shape[1] <= 3:
        buml_obj.vis.data(X, labelsMat[:,0], figsize=figsize1, s=s1, cmap='hsv', title='data_hsv')
    else:
        print('Cannot plot because input data has more than 3 features.')
    if X.shape[1] <= 3:
        buml_obj.vis.data(X, labelsMat[:,0], figsize=figsize1, s=s1, cmap='summer', title='data_summer')
    else:
        print('Cannot plot because input data has more than 3 features.')

    print('Eigenvalues')
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        buml_obj.vis.eigenvalues(buml_obj.LocalViews.GL.lmbda, figsize=figsize1)
    else:
        print('Local views were constructed using', local_algo)

    print('#'*50, flush=True)
    print('Eigenvectors on data', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        n_eigvevs = 3
        for k in range(n_eigvevs):
            if X.shape[1] <= 3:
                buml_obj.vis.eigenvector(X, buml_obj.LocalViews.GL.phi, k, figsize=figsize1, s=50)
            else:
                print('Cannot plot because input data has more than 3 features.')
    else:
        print('Local views were constructed using', local_algo)
        
    print('#'*50, flush=True)
    print('Eigenvectors on embedding', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        n_eigvevs = 3
        for k in range(n_eigvevs):
            if buml_obj.d <= 3:
                buml_obj.vis.eigenvector(buml_obj.GlobalViews.y_final, buml_obj.LocalViews.GL.phi, k, figsize=figsize1, s=s1)
            else:
                print('Cannot plot because embedding dim > 3.')
    else:
        print('Local views were constructed using', local_algo)
        
    
    print('#'*50, flush=True)
    print('gamma on  data', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            n_eigvevs = 3
            for k in range(n_eigvevs):
                if X.shape[1] <= 3:
                    buml_obj.vis.gamma(X, buml_obj.LocalViews.gamma,
                               int(k*buml_obj.local_opts['N']/n_eigvevs),
                               figsize=figsize1, s=50)
                else:
                    print('Cannot plot because input data has more than 3 features.')
        else:
            print('buml_obj.debug is False, thus buml_obj.LocalViews.gamma is not saved.')
    else:
        print('Local views were constructed using', local_algo)

    print('#'*50, flush=True)
    print('gamma on embedding', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            n_eigvevs = 3
            for k in range(n_eigvevs):
                if buml_obj.d <= 3:
                    buml_obj.vis.gamma(buml_obj.GlobalViews.y_final, buml_obj.LocalViews.gamma,
                               int(k*buml_obj.local_opts['N']/n_eigvevs),
                               figsize=figsize1, s=50)
                else:
                    print('Cannot plot because embedding dim > 3.')
        else:
            print('buml_obj.debug is False, thus buml_obj.LocalViews.gamma is not saved.')
    else:
        print('Local views were constructed using', local_algo)

    
    print('#'*50, flush=True)
    print('No. of eigenvectors with small gradients at each point - possibly identifies boundary', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            if X.shape[1] <= 3:
                for thresh in threshs:
                    buml_obj.vis.n_eigvecs_w_grad_lt(X, buml_obj.LocalViews.IPGE.Atilde,
                                                 thresh_prctile=thresh, figsize=figsize2)
            else:
                print('Cannot plot because input data has more than 3 features')
        else:
            print('buml_obj.debug is False, thus buml_obj.LocalViews.IPGE.Atilde is not saved.')
    else:
        print('Local views were constructed using', local_algo)

    print('#'*50, flush=True)
    print('Same visualization as above but plots based on the embedding', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            if buml_obj.d <= 3:
                for thresh in threshs:
                    buml_obj.vis.n_eigvecs_w_grad_lt(buml_obj.GlobalViews.y_final,
                                                 buml_obj.LocalViews.IPGE.Atilde,
                                                 thresh_prctile=thresh, figsize=figsize2)
            else:
                print('Cannot plot because embedding dim > 3')
        else:
            print('buml_obj.debug is False, thus buml_obj.LocalViews.IPGE.Atilde is not saved.')
    else:
        print('Local views were constructed using', local_algo)

    print('#'*50, flush=True)
    print('Distortion of local parameterizations without post-processing', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if buml_obj.debug:
        buml_obj.vis.distortion_boxplot(np.log(buml_obj.LocalViews.local_param_pre.zeta),
                                    title='log(distortion) without postprocessing',
                                    figsize=figsize1)
    else:
        print('buml_obj.debug is False, thus buml_obj.LocalViews.local_param_pre is not saved.')
        
    if buml_obj.d <= 3:
        buml_obj.vis.distortion(buml_obj.GlobalViews.y_final, buml_obj.LocalViews.local_param_post.zeta,
                            'Embedding colored by distortion',
                            figsize=(8,8), s=50)

    print('#'*50, flush=True)
    print('Distortion of local parameterizations with post-processing', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    buml_obj.vis.distortion_boxplot(np.log(buml_obj.LocalViews.local_param_post.zeta),
                                    title='log(distortion)',
                                    figsize=figsize1)
    if X.shape[1] <= 3:
        buml_obj.vis.distortion(X, buml_obj.LocalViews.local_param_post.zeta,
                            'Data colored by distortion',
                            figsize=(8,8), s=50)
    else:
        print('Cannot plot because input data has more than 3 features')
    if buml_obj.d <= 3:
        buml_obj.vis.distortion(buml_obj.GlobalViews.y_final, buml_obj.LocalViews.local_param_post.zeta,
                            'Embedding colored by distortion',
                            figsize=(8,8), s=50)
    else:
        print('Cannot plot because embedding dim > 3')

    print('#'*50, flush=True)
    print('Here we visualize:')
    print('1. Local views in the ambient and embedding space.')
    print('2. Chosen eigenvectors to construct the local parameterization.')
    print('3. Deviation of the chosen eigenvectors from being orthogonal and having same length.', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            if X.shape[1] <= 3:
                for k_ in range(n_views):
                    if interact:
                        k = None
                    else:
                        k = int(k_*X.shape[0]/n_views)
                    buml_obj.vis.local_views(X, buml_obj.LocalViews.local_param_post, buml_obj.LocalViews.U.toarray(),
                                         buml_obj.LocalViews.gamma, buml_obj.LocalViews.IPGE.Atilde,
                                         k=k, figsize=figsize3, save_subdir='data_space')
            else:
                print('Cannot plot because input data has more than 3 features')
        else:
            print('buml_obj.debug is False, thus intermediary data was not saved.')
    else:
        if buml_obj.debug:
            if X.shape[1] <= 3:
                for k_ in range(n_views):
                    if interact:
                        k = None
                    else:
                        k = int(k_*X.shape[0]/n_views)
                    buml_obj.vis.local_views_lpca(X, buml_obj.LocalViews.local_param_post, 
                                              buml_obj.LocalViews.U.toarray(),
                                               k=k, figsize=figsize3, save_subdir='data_space')
            else:
                print('Cannot plot because input data has more than 3 features')
        else:
            print('buml_obj.debug is False, thus intermediary data was not saved.')

    print('#'*50, flush=True)
    print('Same visualization as above but plots based on the embedding.', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            if buml_obj.d <= 3:
                for k_ in range(n_views):
                    if interact:
                        k = None
                    else:
                        k = int(k_*X.shape[0]/n_views)
                    buml_obj.vis.local_views(buml_obj.GlobalViews.y_final, buml_obj.LocalViews.local_param_post,
                                         buml_obj.LocalViews.U.toarray(),
                                         buml_obj.LocalViews.gamma, buml_obj.LocalViews.IPGE.Atilde,
                                         k=k, figsize=figsize3, save_subdir='embedding_space')
            else:
                print('Cannot plot because input data has more than 3 features')
        else:
            print('buml_obj.debug is False, thus intermediary data was not saved.')
    else:
        if buml_obj.debug:
            if buml_obj.d <= 3:
                for k_ in range(n_views):
                    if interact:
                        k = None
                    else:
                        k = int(k_*X.shape[0]/n_views)
                    buml_obj.vis.local_views_lpca(buml_obj.GlobalViews.y_final,
                                              buml_obj.LocalViews.local_param_post,
                                              buml_obj.LocalViews.U.toarray(),
                                              k=k, figsize=figsize3, save_subdir='embedding_space')
            else:
                print('Cannot plot because input data has more than 3 features')
        else:
            print('buml_obj.debug is False, thus intermediary data was not saved.')

    print('#'*50, flush=True)
    print('Chosen eigenvectors indices for local views', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if (X.shape[1] <= 3) and (buml_obj.d ==2):
            buml_obj.vis.chosen_eigevec_inds_for_local_views(X,
                                                         buml_obj.LocalViews.local_param_post.Psi_i,
                                                         figsize=figsize2)
        else:
            print('Cannot plot because input data has more than 3 features or embedding dim > 2')
    else:
        print('Local views were constructed using', local_algo)
        
    print('#'*50, flush=True)
    print('Same visualization but plots based on embedding', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.d == 2:
            buml_obj.vis.chosen_eigevec_inds_for_local_views(buml_obj.GlobalViews.y_final,
                                                         buml_obj.LocalViews.local_param_post.Psi_i,
                                                         figsize=figsize2)
        else:
            print('Cannot plot because embedding dim > 2')
    else:
        print('Local views were constructed using', local_algo)

    print('Sequence of intermediate views', flush=True)
    print('#'*50, flush=True)
    if buml_obj.debug:
        if (buml_obj.d <= 3) and (len(buml_obj.GlobalViews.seq_of_intermed_views_in_cluster) == 1):
            buml_obj.vis.seq_of_intermediate_views(buml_obj.GlobalViews.y_final, buml_obj.IntermedViews.c,
                                               buml_obj.GlobalViews.seq_of_intermed_views_in_cluster[0],
                                               buml_obj.GlobalViews.parents_of_intermed_views_in_cluster[0],
                                               buml_obj.IntermedViews.Utilde, figsize=(8,8), s=50, cmap='tab20')
        else:
            print('Cannot plot because embedding dim > 3')
    else:
        print('buml_obj.debug is False, thus intermediary data was not saved.')

    print('#'*50, flush=True)
    print('Distortion of intermediate views', flush=True)
    print('#'*50, flush=True)
    if X.shape[1] <= 3:
        buml_obj.vis.distortion(X,
                            buml_obj.IntermedViews.intermed_param.zeta[buml_obj.IntermedViews.c],
                            'Distortion of Intermediate Views', figsize=(8,8), s=50)
    else:
        print('Cannot plot because embedding dim > 3')
    if buml_obj.d <= 3:
        buml_obj.vis.distortion(buml_obj.GlobalViews.y_final,
                            buml_obj.IntermedViews.intermed_param.zeta[buml_obj.IntermedViews.c],
                            'Distortion of Intermediate Views', figsize=(8,8), s=50)
    else:
        print('Cannot plot because embedding dim > 3')
    buml_obj.vis.distortion_boxplot(np.log(buml_obj.IntermedViews.intermed_param.zeta),
                                title='log(distortion of intermediate views)',
                                figsize=figsize1)

    print('#'*50, flush=True)
    print('Here we visualize: 1. Intermediate views in the ambient and embedding space.')
    print('2. Chosen eigenvectors to construct the intermediate parameterization.')
    print('3. Deviation of the chosen eigenvectors from being orthogonal and having same length.', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            if X.shape[1] <= 3:
                for k in range(n_views):
                    buml_obj.vis.intermediate_views(X, buml_obj.LocalViews.GL.phi, buml_obj.IntermedViews.Utilde,
                                                buml_obj.LocalViews.gamma, buml_obj.LocalViews.IPGE.Atilde,
                                                buml_obj.IntermedViews.intermed_param.Psi_gamma,
                                                buml_obj.IntermedViews.intermed_param.Psi_i,
                                                buml_obj.IntermedViews.intermed_param.zeta,
                                                buml_obj.IntermedViews.c, k=int(X.shape[0]*k/n_views),
                                                figsize=figsize3, save_subdir='data_space')
            else:
                print('Input has more than 3 features')
        else:
            print('buml_obj.debug is False, thus intermediary data was not saved.')
    else:
        print('Local views were constructed using', local_algo)

    print('#'*50, flush=True)
    print('Same visualization as above but plots based on the embedding', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.debug:
            if buml_obj.d <= 3:
                for k in range(n_views):
                    buml_obj.vis.intermediate_views(buml_obj.GlobalViews.y_final, buml_obj.LocalViews.GL.phi,
                                                buml_obj.IntermedViews.Utilde,
                                                buml_obj.LocalViews.gamma, buml_obj.LocalViews.IPGE.Atilde,
                                                buml_obj.IntermedViews.intermed_param.Psi_gamma,
                                                buml_obj.IntermedViews.intermed_param.Psi_i,
                                                buml_obj.IntermedViews.intermed_param.zeta,
                                                buml_obj.IntermedViews.c, k=int(X.shape[0]*k/n_views),
                                                figsize=figsize3, save_subdir='embedding_space')
            else:
                print('Cannot plot because embedding dim > 3')
        else:
            print('buml_obj.debug is False, thus intermediary data was not saved.')
    else:
        print('Local views were constructed using', local_algo)

    print('#'*50, flush=True)
    print('Chosen eigenvectors indices for intermediate views', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if (X.shape[1]<=3) and (buml_obj.d == 2):
            buml_obj.vis.chosen_eigevec_inds_for_intermediate_views(X,
                                                                buml_obj.IntermedViews.intermed_param.Psi_i,
                                                                buml_obj.IntermedViews.c, figsize=figsize2)
        else:
            print('Cannot plot because input data has more than 3 features or embedding dim > 2')
    else:
        print('Local views were constructed using', local_algo)
    
    print('#'*50, flush=True)
    print('Same visualization but plots based on embedding', flush=True)
    print('#'*50, flush=True)
    local_algo = buml_obj.local_opts['algo']
    if local_algo == 'LDLE':
        if buml_obj.d == 2:
            buml_obj.vis.chosen_eigevec_inds_for_intermediate_views(buml_obj.GlobalViews.y_final,
                                                                buml_obj.IntermedViews.intermed_param.Psi_i,
                                                                buml_obj.IntermedViews.c, figsize=figsize2)
        else:
            print('Cannot plot because embedding dim > 2')
    else:
        print('Local views were constructed using', local_algo)

    print('initial global embedding', flush=True)
    print('#'*50, flush=True)
    if buml_obj.d <= 3:
        buml_obj.vis.global_embedding(buml_obj.GlobalViews.y_init, buml_obj.vis_opts['c'], buml_obj.vis_opts['cmap_interior'],
                                  buml_obj.GlobalViews.color_of_pts_on_tear_init, buml_obj.vis_opts['cmap_boundary'],
                              'Initial Embedding', figsize=figsize1, s=50)
    else:
        print('Cannot plot because embedding dim > 3')

    print('#'*50, flush=True)
    print('final global embedding', flush=True)
    print('#'*50, flush=True)
    if buml_obj.d <= 3:
        buml_obj.vis.global_embedding(buml_obj.GlobalViews.y_final, buml_obj.vis_opts['c'], 'hsv',
                                  buml_obj.GlobalViews.color_of_pts_on_tear_final, buml_obj.vis_opts['cmap_boundary'],
                                  'Final Embedding', figsize=figsize1, s=50)
    else:
        print('Cannot plot because embedding dim > 3')
    
    if 'puppets' in fpath:
        X, labelsMat, buml_obj, img, img_shape = all_data[:5]
        buml_obj.vis.global_embedding_images_v2(img, img_shape[::-1], buml_obj.GlobalViews.y_final, labelsMat[:,0]*0,
                                    buml_obj.vis_opts['cmap_interior'], buml_obj.GlobalViews.color_of_pts_on_tear_final,
                                    buml_obj.vis_opts['cmap_boundary'], 'images',
                                    offset_ratio=0.2, zoom=0.4, nx=8, ny=10, v_ratio=0.65, w_ratio=0.005,
                                    figsize=(10,12), s=80, to_remove=False, k_to_avoid=[], to_T=True)
        buml_obj.vis.global_embedding(buml_obj.GlobalViews.y_final, X[:,0], 'summer',
                                  buml_obj.GlobalViews.color_of_pts_on_tear_final, buml_obj.vis_opts['cmap_boundary'],
                                  'Final Embedding 2', figsize=(5,5), s=50)
    
    if 'epoch' in fpath:
        Ls = all_data[4]
        buml_obj.vis.visualize_epoch_data(buml_obj.GlobalViews.y_final, Ls)