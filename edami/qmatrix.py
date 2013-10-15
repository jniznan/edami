from __future__ import division
import numpy as np
from utils import nan_rmse


def sgd_(X, alpha, concepts, n_iters, qmatrix, skills, b, learn_b=True):
    S = X.shape[0]
    P = X.shape[1]
    pairs = [(s, p) for s in range(S) for p in range(P)
             if not np.isnan(X[s, p])]
    for _ in range(n_iters):
        np.random.shuffle(pairs)
        for s, p in pairs:
            prediction = b[p] + np.dot(skills[s], qmatrix[:, p])
            error = X[s, p] - prediction
            k = -alpha * error
            if learn_b:
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


try:
    # try to import cythonized version:
    from sgd import sgd
except ImportError:
    sgd = sgd_


class QMatrix(object):

    def __init__(self, concepts=2, n_iters=30, alpha=0.005):
        self.concepts = concepts
        self.n_iters = n_iters
        self.alpha = alpha

    def fit(self, X, qmatrix=None, learn_b=True):
        """ Normalization: want elements of Q-matrix be betw. 0 - 1 """
        S, P = X.shape
        C = self.concepts
        if qmatrix is None:
            qmatrix = np.random.normal(loc=0.5, scale=0.1, size=(C, P))
        skills = np.random.normal(scale=0.1, size=(S, C))
        mdat = np.ma.masked_array(X, np.isnan(X))
        b = np.mean(mdat, axis=0).filled(0) if learn_b else np.zeros(P)

        self.qmatrix, self.skills, self.b = sgd(X, self.alpha, int(C),
                                                self.n_iters, qmatrix,
                                                skills, b, learn_b)
        self.prediction = np.dot(self.skills, self.qmatrix) + self.b
        return nan_rmse(X, self.prediction)
