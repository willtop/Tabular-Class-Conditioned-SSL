import numpy as np
from sklearn.cluster import SpectralClustering

class RandomMaskGenerator():
    def __init__(self, n_features, corruption_rate):
        self.n_features = n_features
        self.corruption_rate = corruption_rate
        self.corruption_len = int(np.ceil(self.corruption_rate * self.n_features))
        assert self.corruption_len < self.n_features
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for i in range(n_samples):
            corruption_idxes = np.random.permutation(self.n_features)[:self.corruption_len]
            corruption_masks[i, corruption_idxes] = True
        return corruption_masks
    

class CorrelationMaskGenerator(RandomMaskGenerator):
    def __init__(self, n_features, corruption_rate, high_correlation):
        super().__init__(n_features, corruption_rate)
        self.high_correlation = high_correlation
        self.softmax_temporature = 0.3

    def initialize_probabilities(self, feat_impt):
        assert np.shape(feat_impt) == (self.n_features, self.n_features)
        # convert into probabilities
        feat_impt_prob_tmp = np.exp(feat_impt/self.softmax_temporature)
        feat_impt_prob_tmp = feat_impt_prob_tmp * (1-np.eye(self.n_features))
        self.feat_impt_prob = feat_impt_prob_tmp / np.sum(feat_impt_prob_tmp, axis=1, keepdims=True)
        if not self.high_correlation:
            for i in range(self.n_features):
                # simply flip the indices and reassign the probability values
                feat_impt_prob_tmp = np.delete(self.feat_impt_prob[i], obj=i)
                feat_impt_prob_tmp = np.sort(feat_impt_prob_tmp)[::-1][np.argsort(np.argsort(feat_impt_prob_tmp))]
                self.feat_impt_prob[i] = np.insert(feat_impt_prob_tmp, obj=i, values=0)
        return 
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for i in range(n_samples):
            selected_idxes = []
            remaining_idxes = np.arange(self.n_features)
            selected_id = np.random.choice(self.n_features)
            selected_idxes.append(selected_id)
            remaining_idxes = np.delete(remaining_idxes, obj=selected_id)
            for _ in range(1, self.corruption_len):
                sampling_prob_onerow_tmp = self.feat_impt_prob[selected_idxes]
                sampling_prob_onerow_tmp = sampling_prob_onerow_tmp[:,remaining_idxes]
                # consider the weakest (or strongest) link of each feature to features already selected
                sampling_prob_onerow = np.min(sampling_prob_onerow_tmp, axis=0)
                selected_id = np.random.choice(remaining_idxes, p=sampling_prob_onerow/np.sum(sampling_prob_onerow))
                selected_idxes.append(selected_id)
                remaining_idxes = np.delete(remaining_idxes, np.where(remaining_idxes==selected_id))
            assert len(selected_idxes) == self.corruption_len
            corruption_masks[i, selected_idxes] = True
        return corruption_masks    