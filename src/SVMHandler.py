# Wrapper for SVM Handling

from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np

# Possible Open Items:
# - make class for easy kernel switching


class SVMHandler:

    def __init__(self, kernel='linear'):

        if kernel == 'ncc':
            self.kernel = kernel
            self.kernel = self.ncc
        else:
            self.kernel = kernel
            self.kernel_method = self.none

    def precompute(self, x, y):
        return self.kernel(x,y)

    def none(self, x, y):
        return x

    def ncc(self, x, y):
        x -= np.mean(x)
        y -= np.mean(y)
        x /= np.sqrt(np.mean(np.sum(np.power(x, 2))))
        y /= np.sqrt(np.mean(np.sum(np.power(y, 2))))
        return np.dot(x, y.T)

    def grid_cv_optimized_svm(self, data_train):

        # Build Hyperspace for Parameter search.
        c_s = np.logspace(-6, 9, 15)
        gamma = np.logspace(-15, 3, 15)

        # Initializing parameterless svm
        svc = svm.SVC(kernel=self.kernel)

        # Extensive Search of the Paramater Hyperspace for the C and gamma parameter.
        clf = GridSearchCV(estimator=svc, param_grid=dict(C=c_s, gamma=gamma), n_jobs=1)
        clf.fit(X=self.kernel_method(data_train[:, 1:], data_train[:, 1:]), y=data_train[:, 0])

        return clf.best_estimator_
