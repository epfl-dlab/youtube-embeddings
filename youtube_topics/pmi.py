#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Taken from https://github.com/asahala/pmi-embeddings/blob/main/src/make_embeddings.py
"""

import itertools
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

__version__ = "2021-01-26"

# Globals. All meta-symbols must have a string and integer form.
# Integers must be negative!
PADDING = {"str": "#", "int": -3}
STOP = {"str": "<stop>", "int": -2}
LACUNA = {"str": "_", "int": -1}

# If you use more symbols, add them here as well in their string form.
META_SYMBOLS = (PADDING["str"], LACUNA["str"], STOP["str"])

# Pretty print dividers
DIV = "> " + "--" * 24


def calculate_pmi(
    cooc_matrix,
    shift_type=0,
    alpha=None,
    lambda_=None,
    threshold=0,
    pmi_variant=None,
    scale=1,
):
    """Calculate Shifted PMI matrix with various modifications
    :param lambda_                  Dirichlet smoothing
    :param alpha                    Context distribution smoothing
    :param threshold                PMI shift value k (see formulae below)
    :param shift_type               set PMI shift type
                                      0: Jungmaier et al. 2020
                                      1: Levy & Goldberg 2014
                                      2: Experimental: linear addition
    :param variant                  set PMI variant
    :type lambda_                   float (recommended: 0.0001)
    :type alpha                     float (recommended: 0.75)
    :type threshold                 integer (useful values: 0-10)
    :type shift_type                integer (0, 1, 2)
                                      0: if PMI(a,b) < -k, PMI(a,b) = 0
                                      1: max(PMI(a,b) - log2(k), 0)
                                      2: max(PMI(a,b) + k, 0)
    :type variant                   str (pmi2, npmi)
    For PMI(*,*) it is more efficient to calculate the scores using
    elementwise matrix multiplication thanks to optimization in
    numpy/scipy as in Levy et al. 2015. For each element (a,b) in the
    matrix:
                     p(a,b)                               1
    PMI(a,b) = log2 --------   =  log2 ( N * f(a,b) * -------- )
                    p(a)p(b)                          f(a)f(b)
    """

    sum_a = np.array(cooc_matrix.sum(axis=1))[:, 0]
    sum_b = np.array(cooc_matrix.sum(axis=0))[0, :]

    """ Dirichlet and context distribution smoothing as in
       Jacob Jungmaier (Accessed: 2020-12-01):
          https://github.com/jungmaier/dirichlet-smoothed-word-embeddings/
       Omer Levy (Accessed: 2019-05-30)
          https://bitbucket.org/omerlevy/hyperwords/ """
    if lambda_ is not None:
        sum_a += lambda_ * cooc_matrix.shape[0]
        sum_b += lambda_ * cooc_matrix.shape[0]
        # TODO: do not modify the cooc_matrix directly! May cause
        #       problems if someone runs this method several times
        #       without recalculating the matrix!
        cooc_matrix.data = cooc_matrix.data + lambda_
    if alpha is not None:
        sum_b = sum_b**alpha

    sum_total = sum_b.sum()
    pmi = csr_matrix(cooc_matrix)

    """ Scale co-oc frequencies by window size; this has to
    be done after row and column summation """
    if scale != 1:
        pmi *= scale

    """ Calculate PMI (take reciprocals and multiply all
    columns and rows with reciprocal * sum to get the
    products of marginal probabilities. """
    with np.errstate(divide="ignore"):
        sum_a = np.reciprocal(sum_a)
        sum_b = np.reciprocal(sum_b)

    pmi = pmi.multiply(sum_a[:, None]).multiply(sum_b[:, None].T) * sum_total
    pmi.data = np.log2(pmi.data)

    """ Various PMI derivations:

    NPMI          (Gerlof Bouma 2009): PMI / -log2 p(a,b)
    PMI^2         (Daille 1994): I define PMI^2 here as 
                  PMI - ((1+x) * -log2 p(a,b)), where x is a small smoothing
                  factor to make sure perfect dependencies are not
                  confused with null co-occurrences in the sparse matrix, 
                  as PMI^2 has bounds of 0 > log2 p(a,b) > -inf. x = 0.0001
                  should be enough for corpora of few million words 
                  to avoid any bigram getting a score of 0.0 """

    if pmi_variant == "npmi":
        joint_dist_matrix = cooc_matrix.data * (1 / sum_total)
        pmi.data = pmi.data / -np.log2(joint_dist_matrix.data)
    elif pmi_variant == "pmi2":
        joint_dist_matrix = cooc_matrix.data * (1 / sum_total)
        pmi.data -= 1.0001 * -np.log2(joint_dist_matrix.data)

    """ Apply threshold for Shifted PMI. """
    if threshold is not None:
        if shift_type == 0:
            pmi.data[pmi.data < -threshold] = 0
        elif shift_type == 1:
            pmi.data *= np.log2(threshold)
            pmi.data[pmi.data < 0] = 0
        elif shift_type == 2:
            pmi.data += threshold
            pmi.data[pmi.data < 0] = 0

    return pmi
