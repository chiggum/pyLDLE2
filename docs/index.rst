.. pyLDLE2 documentation master file, created by
   sphinx-quickstart on Thu Jul 21 09:29:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyLDLE2's documentation!
===================================

Installation
============

``pip install pyLDLE2``


API
====

The main LDLE class is as follows.

.. autoclass:: pyLDLE2.ldle_.LDLE
    :members:

   
**Hyperparameters**: A description of all the available hyperparameters and their default values are provided below.

.. autoclass:: pyLDLE2.ldle_.get_default_local_opts
.. autoclass:: pyLDLE2.ldle_.get_default_intermed_opts
.. autoclass:: pyLDLE2.ldle_.get_default_global_opts
.. autoclass:: pyLDLE2.ldle_.get_default_vis_opts

Modules
==================
An index of the available modules and the functions within them.

* :ref:`modindex`

Example
========

.. include:: example/example.rst
