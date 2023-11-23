import numpy as np

class RandomMask():
    def __init__(self, n_features, corruption_rate):
        self.n_features = n_features
        self.corruption_rate = corruption_rate
        self.corruption_len = int(self.corruption_rate * self.n_features)
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for k in range(n_samples):
            corruption_idxes = np.random.permutation(self.n_features)[:self.corruption_len]
            corruption_masks[k, corruption_idxes] = True
        return corruption_masks
    

class CrossClusterMask(RandomMask):
    def __init__(self, n_features, corruption_rate):
        super().__init__(n_features, corruption_rate)

    def _spectrum_clustering(self, cov_mat):
        return

    def fit_feature_clusters(self, training_data):
        cov_mat = np.cov(training_data)
        return
    
    def get_masks(self, n_samples):
        return 
