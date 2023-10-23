import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

# As a base class with random corruption, does not need targets
class RandomCorruptSampler(Dataset):
    def __init__(self, data, batch_size):
        assert isinstance(data, pd.DataFrame)
        self.data = np.array(data,dtype='object') 
        self.batch_size = batch_size
        self.columns = data.columns
        self.n_samples = np.shape(self.data)[0]
        self.n_batches = int(np.ceil(self.n_samples/batch_size))
        self._initialize_epoch()

    def _initialize_epoch(self):
        np.random.shuffle(self.data)
        self.end_pointer = 0
        return

    def __len__(self):
        return np.shape(self.data)[0]
                             
    def _get_one_sample_pair(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        sample = self.data[index]

        random_idx = np.random.randint(0, len(self))
        random_sample = self.data[random_idx]

        return sample, random_sample

    def sample_batch(self):
        anchors, random_samples = [], []
        new_end_pointer = min(self.end_pointer + self.batch_size, self.n_samples)
        for i in range(self.end_pointer, new_end_pointer):
            anchor, random_sample = self._get_one_sample_pair(i)
            anchors.append(anchor)
            random_samples.append(random_sample)
        if new_end_pointer == self.n_samples:
            self._initialize_epoch()
        else:
            self.end_pointer = new_end_pointer
        return np.array(anchors, dtype='object'), np.array(random_samples, dtype='object')

    def get_data(self):
        return self.data
    
    def get_data_columns(self):
        return self.columns

    @property
    def shape(self):
        return self.data.shape
    

# Can be used with both predicted classes: bootstrapping from semi-supervised learning;
# or with oracle class labels
class ClassCorruptSampler(RandomCorruptSampler):
    def __init__(self, data, batch_size, target):
        super().__init__(data, batch_size)
        self.target = target

    def _get_one_sample_pair(self, index):
        # Modification: the sample used to corrupt the anchor has to be from the same class
        sample = self.data[index]

        candidate_idxes = np.where(self.target == self.target[index])[0]
        random_idx = np.random.choice(candidate_idxes)
        random_sample = self.data[random_idx]
        
        return sample, random_sample