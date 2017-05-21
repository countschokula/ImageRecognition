# Wrapper for SVM Handling

from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np

# Possible Open Items:
# - ncc_kernel


def ncc(x, y):
    x -= np.mean(x)
    y -= np.mean(y)
    x /= np.sqrt(np.mean(np.sum(np.power(x, 2))))
    y /= np.sqrt(np.mean(np.sum(np.power(y, 2))))
    return np.dot(x, y.T)


def grid_cv_optimized_svm(data_train, kernel='rbf'):

    # Build Hyperspace for Parameter search.
    c_s = np.logspace(-6, 9, 15)
    gamma = np.logspace(-15, 3, 15)

    # Initializing parameterless svm
    svc = svm.SVC(kernel=kernel)

    # Extensive Search of the Paramater Hyperspace for the C and gamma parameter.
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=c_s, gamma=gamma), n_jobs=1)
    clf.fit(X=data_train[:, 1:], y=data_train[:, 1])

    return clf.best_estimator_
