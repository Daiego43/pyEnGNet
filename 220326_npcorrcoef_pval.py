__author__ = "Fernando M. Delgado-Chaves"
__email__ = "fernando.miguel.delgado-chaves@hamburg-uni.de"

# Author:           Fernando M. Delgado-Chaves
# Affiliation:      Chair of Computational Systems Biology, University of Hamburg, Germany
# Credits:          Fernando M. Delgado-Chaves
# Email:            fernando.miguel.delgado-chaves@hamburg-uni.de
# Created Date:     26/03/22
# Version:          1.0

import numpy as np
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
import scipy.special as sc


def corrcoef_pval(matrix):
    # here we calculate pearson's r normally using corrcoef (fastest method)
    r = np.corrcoef(matrix)
    # r[np.diag_indices(r.shape[0])] = 0
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = sc.betainc(0.5 * df, 0.5, df / (df + ts))

    # Then we estimate pvalues which indicate how different is the correlation coefficient from 0.
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

    # Then we correct for multiple testing using holm method (Very restrictive, maybe set this as an arugment for the
    # user). This adj. pvalue will be used for filtering later on.
    padj = np.zeros(shape=r.shape)
    padj[np.triu_indices(padj.shape[0], 1)] = multipletests(pf, method="holm")[1]
    padj[np.tril_indices(padj.shape[0], -1)] = padj.T[np.tril_indices(padj.shape[0], -1)]
    padj[np.diag_indices(padj.shape[0])] = np.ones(padj.shape[0])

    # min(padj)
    return r, p, padj


def corrcoef_loop(matrix):
    # Old solution that iterates through all posible combinations. Still the only one available for very large
    # matrices (> 25000 features).
    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i + 1, rows):
            r_, p_ = ss.pearsonr(matrix[i], matrix[j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


dummy = np.random.rand(50, 200)  # genes and 200 samples.

r0, p0, p0adj = corrcoef_pval(dummy)
r1, p1 = corrcoef_loop(dummy)

print(r0 == r1)  # If we round the result we see it's the same.
print(p0 == p1)
