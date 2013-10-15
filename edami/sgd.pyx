# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
import sys
from time import time

cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER


def sgd(np.ndarray[DOUBLE, ndim=2, mode='c'] X,
        double alpha,
        Py_ssize_t concepts,
        Py_ssize_t n_iters,
        np.ndarray[DOUBLE, ndim=2, mode='c'] qmatrix,
        np.ndarray[DOUBLE, ndim=2, mode='c'] skills,
        np.ndarray[DOUBLE, ndim=1, mode='c'] b,
        Py_ssize_t learn_b):

    cdef int S = X.shape[0]
    cdef int P = X.shape[1]
    cdef np.ndarray[long, ndim=1, mode='c'] pairs
    cdef double prediction = 0.0
    cdef double error = 0.0
    cdef double k = 0.0
    cdef int ele = 0
    cdef int s = 0
    cdef int p = 0

    pairs = np.array([s + p*S for s in range(S) for p in range(P) if not np.isnan(X[s, p])])
    for epoch in range(n_iters):
        np.random.shuffle(pairs)
        for ele in pairs:
            s = ele % S
            p = ele / S
            prediction = b[p] + np.dot(skills[s], qmatrix[:, p])
            error = X[s, p] - prediction
            k = -alpha * error
            if learn_b == 1:
                b[p] += -k
            for c in range(concepts):
                oldskill = skills[s, c]
                skills[s, c] += k * (-qmatrix[c, p])
                qmatrix[c, p] += k * (-oldskill)
                if qmatrix[c, p] < 0:
                    qmatrix[c, p] = 0
                if qmatrix[c, p] > 1:
                    qmatrix[c, p] = 1
    return qmatrix, skills, b
