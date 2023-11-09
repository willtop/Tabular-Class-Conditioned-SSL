import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

class BasicSampler():
    def __init__(self, data, batch_size, target):
        assert isinstance(data, pd.DataFrame)
        self.data = np.array(data,dtype='object') 
        self.target = target
        self.batch_size = batch_size
        self.columns = data.columns
        self.n_samples = np.shape(self.data)[0]
        self.n_batches = int(np.ceil(self.n_samples/batch_size))
        self.batch_end_pointer = 0

    def __len__(self):
        return np.shape(self.data)[0]

    def __str__(self):
        return f"A BasicSampler object, with datasize {len(self)}."

    def _initialize_epoch(self):
        perm = np.random.permutation(len(self))
        self.data = self.data[perm]
        # shuffle the target to align with data for class-based sampler
        if isinstance(self.target, np.ndarray):
            self.target = self.target[perm]
        self.batch_end_pointer = 0
        return

    # To be overwritten by subclasses for its own needs
    def _get_one_sample_pair(self):
        raise NotImplementedError
    
    def sample_batch(self):
        data_batch_1, data_batch_2 = [], []
        new_batch_end_pointer = min(self.batch_end_pointer + self.batch_size, self.n_samples)
        for i in range(self.batch_end_pointer, new_batch_end_pointer):
            data_1, data_2 = self._get_one_sample_pair(i)
            data_batch_1.append(data_1)
            data_batch_2.append(data_2)
        if new_batch_end_pointer == self.n_samples:
            self._initialize_epoch()
        else:
            self.batch_end_pointer = new_batch_end_pointer
    
        return np.array(data_batch_1, dtype='object'), np.array(data_batch_2, dtype='object')
    
    def get_data(self):
        return self.data
    
    def get_data_columns(self):
        return self.columns
    
    
    @property
    def shape(self):
        return self.data.shape


# As a base class with random corruption, does not need targets
class RandomCorruptSampler(BasicSampler):
    def __init__(self, data, batch_size, target=None):
        super().__init__(data, batch_size, target)        

    def __str__(self):
        return f"A RandomCorruptSampler object, with datasize {len(self)}."
                             
    def _get_one_sample_pair(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        sample = self.data[index]

        random_idx = np.random.randint(0, len(self))
        random_sample = self.data[random_idx]

        return sample, random_sample
    


# Can be used with both predicted classes: bootstrapping from semi-supervised learning;
# or with oracle class labels
class ClassCorruptSampler(RandomCorruptSampler):
    def __init__(self, data, batch_size, target):
        super().__init__(data, batch_size, target)

    def __str__(self):
        return f"A ClassCorruptSampler object, with datasize {len(self)}."

    # Modification: the sample used to corrupt the anchor has to be from the same class
    def _get_one_sample_pair(self, index):
        sample = self.data[index]

        candidate_idxes = np.where(self.target == self.target[index])[0]
        random_idx = np.random.choice(candidate_idxes)
        random_sample = self.data[random_idx]
        
        return sample, random_sample
    

    
# Used for supervised learning
class SupervisedSampler(BasicSampler):
    def __init__(self, data, batch_size, target):
        super().__init__(data, batch_size, target)

    def __str__(self):
        return f"A SupervisedSampler object, with datasize {len(self)}."
    
    # Supervised learning setting: sample a (data, target) pair
    def _get_one_sample_pair(self, index):
        data_single = self.data[index]
        target_single = self.target[index]
        return data_single, target_single
    