import numpy as np


class Preprocession:

    def __init__(self, method, patch_size):
        self.patch_size = patch_size
        if method == 'averaging':
            self.method = self.averaging
            self.feature_length = 6
        else:
            self.feature_length = (self.patch_size**2)*3
            self.method = self.none

    def preprocess(self, patch):
        return self.method(patch)

    def none(self, patch):
        return patch[:, :, :3].flatten()

    def averaging(self, patch):
        sigma0 = np.std(patch[:, :, 0])
        sigma1 = np.std(patch[:, :, 1])
        sigma2 = np.std(patch[:, :, 2])
        average0 = np.mean(patch[:, :, 0])
        average1 = np.mean(patch[:, :, 1])
        average2 = np.mean(patch[:, :, 2])
        return np.array([sigma0, average0, sigma1, average1, sigma2, average2])
