import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

# As a base class with random corruption, does not need targets
class RandomCorruptDataset(Dataset):
    def __init__(self, data, columns):
        self.data = np.array(data)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        sample = torch.tensor(self.data[index], dtype=torch.float)

        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape
    

# Can be used with both predicted classes: bootstrapping from semi-supervised learning;
# or with oracle class labels
class ClassCorruptDataset(RandomCorruptDataset):
    def __init__(self, data, target, columns):
        super().__init__(data, columns)
        self.target = target

    def __getitem__(self, index):
        # Modification: the sample used to corrupt the anchor has to be from the same class
        sample = torch.tensor(self.data[index], dtype=torch.float)

        candidate_idxes = np.where(self.target == self.target[index])[0]
        random_idx = np.random.choice(candidate_idxes)
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        
        return sample, random_sample