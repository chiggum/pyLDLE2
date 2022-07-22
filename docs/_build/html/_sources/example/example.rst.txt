.. code:: ipython3

    %matplotlib inline

.. code:: ipython3

    from pyLDLE2 import datasets
    from pyLDLE2 import ldle_


.. parsed-literal::

    matplotlib.get_backend() =  module://matplotlib_inline.backend_inline


.. code:: ipython3

    noise = 0.01

.. code:: ipython3

    save_dir_root = '../data/pyLDLE2/noisyswissroll_'+str(noise)+'/'

.. code:: ipython3

    X, labelsMat, ddX = datasets.Datasets().noisyswissroll(noise=noise)


.. parsed-literal::

    X.shape =  (10260, 3)


.. code:: ipython3

    # The supplied options would override the default options
    ldle = ldle_.LDLE(local_opts={'algo':'LTSA'},
                      vis_opts={'c': labelsMat[:,0]},
                      intermed_opts={'eta_min':5},
                      verbose=True, debug=False)

.. code:: ipython3

    ldle.fit(X=X)


.. parsed-literal::

    Constructing local views using LTSA.
    local_param: 0 points processed...
    local_param: 2565 points processed...
    local_param: 5130 points processed...
    local_param: 7695 points processed...
    local_param: all 10260 points processed...
    Done.
    ##############################
    Time elapsed from last time log: 5.5 seconds
    Total time elapsed: 5.5 seconds
    ##############################
    Max local distortion = 2934.913683389935
    Constructing intermediate views.
    eta = 2.
    # non-empty views with sz < 2 = 10260
    #nodes in views with sz < 2 = 10260
    Costs computed when eta = 2.
    ##############################
    Time elapsed from last time log: 10.3 seconds
    Total time elapsed: 10.4 seconds
    ##############################
    Remaining #nodes in views with sz < 2 = 0
    Done with eta = 2.
    ##############################
    Time elapsed from last time log: 77.4 seconds
    Total time elapsed: 87.8 seconds
    ##############################
    eta = 3.
    # non-empty views with sz < 3 = 270
    #nodes in views with sz < 3 = 540
    Costs computed when eta = 3.
    ##############################
    Time elapsed from last time log: 0.3 seconds
    Total time elapsed: 88.1 seconds
    ##############################
    Remaining #nodes in views with sz < 3 = 0
    Done with eta = 3.
    ##############################
    Time elapsed from last time log: 1.0 seconds
    Total time elapsed: 89.2 seconds
    ##############################
    eta = 4.
    # non-empty views with sz < 4 = 166
    #nodes in views with sz < 4 = 498
    Costs computed when eta = 4.
    ##############################
    Time elapsed from last time log: 0.3 seconds
    Total time elapsed: 89.5 seconds
    ##############################
    Remaining #nodes in views with sz < 4 = 0
    Done with eta = 4.
    ##############################
    Time elapsed from last time log: 1.5 seconds
    Total time elapsed: 91.0 seconds
    ##############################
    eta = 5.
    # non-empty views with sz < 5 = 116
    #nodes in views with sz < 5 = 464
    Costs computed when eta = 5.
    ##############################
    Time elapsed from last time log: 0.2 seconds
    Total time elapsed: 91.2 seconds
    ##############################
    Remaining #nodes in views with sz < 5 = 0
    Done with eta = 5.
    ##############################
    Time elapsed from last time log: 1.8 seconds
    Total time elapsed: 93.1 seconds
    ##############################
    Pruning and cleaning up.
    Done.
    ##############################
    Time elapsed from last time log: 0.3 seconds
    Total time elapsed: 93.4 seconds
    ##############################
    After clustering, max distortion is 13.278407
    Ambiguous overlaps checked for 0 intermediate views
    Ambiguous overlaps checked for 232 intermediate views
    Ambiguous overlaps checked for 464 intermediate views
    Ambiguous overlaps checked for 696 intermediate views
    Ambiguous overlaps checked for 928 points
    Seq of intermediate views and their predecessors computed.
    No. of connected components = 1
    Computing initial embedding using: sequential algorithm
    ##############################
    Time elapsed from last time log: 4.3 seconds
    Total time elapsed: 4.3 seconds
    ##############################
    Initial alignment of 232 views completed
    Initial alignment of 464 views completed
    Initial alignment of 696 views completed
    Embedding initialized.
    ##############################
    Time elapsed from last time log: 0.3 seconds
    Total time elapsed: 4.6 seconds
    ##############################



.. image:: example/output_6_1.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 0
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 14.9 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 24.8 seconds
    Total time elapsed: 39.6 seconds
    ##############################



.. image:: example/output_6_3.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 1
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 44.9 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 26.2 seconds
    Total time elapsed: 71.0 seconds
    ##############################



.. image:: example/output_6_5.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 2
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 76.2 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 26.6 seconds
    Total time elapsed: 102.9 seconds
    ##############################



.. image:: example/output_6_7.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 3
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 108.1 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 27.0 seconds
    Total time elapsed: 135.1 seconds
    ##############################



.. image:: example/output_6_9.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 4
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 140.3 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 27.5 seconds
    Total time elapsed: 167.8 seconds
    ##############################



.. image:: example/output_6_11.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 5
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 173.1 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 28.2 seconds
    Total time elapsed: 201.3 seconds
    ##############################



.. image:: example/output_6_13.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 6
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 206.5 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 27.5 seconds
    Total time elapsed: 234.0 seconds
    ##############################



.. image:: example/output_6_15.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 7
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 239.2 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 29.3 seconds
    Total time elapsed: 268.5 seconds
    ##############################



.. image:: example/output_6_17.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 8
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 273.8 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 27.0 seconds
    Total time elapsed: 300.8 seconds
    ##############################



.. image:: example/output_6_19.png


.. parsed-literal::

    Refining with retraction algorithm for 100 iterations.
    Refinement iteration: 9
    ##############################
    Time elapsed from last time log: 0.0 seconds
    Total time elapsed: 306.0 seconds
    ##############################
    Computing Pseudoinverse of a matrix of L of size 11188
    Descent starts
    Done.
    ##############################
    Time elapsed from last time log: 28.4 seconds
    Total time elapsed: 334.4 seconds
    ##############################
    Computing error.
    Computing Pseudoinverse of a matrix of L of size 11188
    Alignment error: 0.237
    ##############################
    Time elapsed from last time log: 16.0 seconds
    Total time elapsed: 350.3 seconds
    ##############################



.. image:: example/output_6_21.png




.. parsed-literal::

    array([[ 0.82410332, -0.38683067],
           [ 0.81674671, -0.37520528],
           [ 0.81872421, -0.35516005],
           ...,
           [-0.83678387,  0.37806734],
           [-0.84237746,  0.38536084],
           [-0.86196219,  0.38233134]])



.. code:: ipython3

    ldle.vis.distortion(X, ldle.IntermedViews.intermed_param.zeta[ldle.IntermedViews.c],
                        title='Distortion of intermediate views')



.. image:: example/output_7_0.png


.. code:: ipython3

    ldle.GlobalViews.vis_embedding(ldle.GlobalViews.y_final, ldle.vis, ldle.vis_opts,
                                   ldle.GlobalViews.color_of_pts_on_tear_final,
                                   title='Final global embedding', )



.. image:: example/output_8_0.png


