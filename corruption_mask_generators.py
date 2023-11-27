import numpy as np
from sklearn.cluster import SpectralClustering

class RandomMaskGenerator():
    def __init__(self, n_features, corruption_rate):
        self.n_features = n_features
        self.corruption_rate = corruption_rate
        self.corruption_len = int(np.ceil(self.corruption_rate * self.n_features))
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for k in range(n_samples):
            corruption_idxes = np.random.permutation(self.n_features)[:self.corruption_len]
            corruption_masks[k, corruption_idxes] = True
        return corruption_masks
    

class CrossClusterMaskGenerator(RandomMaskGenerator):
    def __init__(self, n_features, corruption_rate):
        super().__init__(n_features, corruption_rate)
        # ensure at least two clusters (so cross-cluster feature selection makes a difference)
        assert self.corruption_len > 1 
        # a spectral cluster model taking in covariance matrix as the affinity matrix
        self.cluster_model = SpectralClustering(n_clusters=self.corruption_len, affinity='precomputed')

    def fit_feature_clusters(self, training_data):
        cov_mat = np.cov(training_data, rowvar=False)
        affinity_mat = np.exp(cov_mat*(1.0-np.eye(cov_mat.shape[0])))
        self.feature_cluster_assignments = self.cluster_model.fit_predict(affinity_mat)
        assert self.feature_cluster_assignments.shape == (self.n_features,)
        assert np.min(self.feature_cluster_assignments) == 0 and np.max(self.feature_cluster_assignments) == self.corruption_len-1
        print("Cross-cluster mask generator fitted complete with feature-cluster assignments.")
        return
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for k in range(n_samples):
            corruption_idxes = []
            # select one feature per cluster, therefore need the same number of clusters 
            # as the number of features to be corrupted
            for i in range(self.corruption_len):
                corruption_idxes.append(np.random.choice(np.where(self.feature_cluster_assignments == i)[0]))
            corruption_masks[k, corruption_idxes] = True
        return corruption_masks 
