import numpy as np
import itertools
from math import factorial

def Hausdorff(e,s):
    # spectral variation
    def _sv(e,s):
        return max([min([abs(e[i]-s[j]) for j in range(len(s))]) for i in range(len(e))])
    # Hausdorff distance
    return max(_sv(e,s),_sv(s,e))

def specR(a): #if this thing is close to 0, then your shit can't be ranked...
    # Set a into the right format
    a[a == -1] = 0
    # given graph Laplacian
    n = len(a)
    x = np.array([np.sum(a[i,:]) for i in range(n)])
    d = np.diag(x)
    l = d - a;
    # perfect dominance graph spectrum and out-degree
    s = np.array([n-k for k in range(1,n+1)])
    # eigenvalues of given graph Laplacian
    e = np.linalg.eigvals(l)
    # rankability measure
    return 1. - ((Hausdorff(e,s)+Hausdorff(x,s))/(2*(n-1)))

def edgeR(a):
    # size
    n = len(a)
    # complete dominance
    domMat = np.triu(1.0*np.ones((n,n)),1)
    # fitness list
    fitness = []
    # brute force (consider all permutations)
    for i in itertools.permutations(range(n)):
        b = a[i,:]
        b = b[:,i]
        # number of edge changes (k) for given permutation
        fitness.append(np.sum(np.abs(domMat - b)))
    # minimum number of edge chagnes
    k = np.amin(fitness)
    # number of permutations that gave this k
    p = np.sum(np.abs(fitness-k)<np.finfo(float).eps)
    # rankability measure
    return 1.0 - 2.0*k*p/(n*(n-1)*factorial(n))