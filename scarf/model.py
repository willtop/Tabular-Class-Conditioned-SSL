import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class Neural_Net(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_depth=4,
        head_depth=2,
        corruption_rate=0.6,
        contrastive_loss_temperature=1.0,
        model_device=None
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.

            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
                encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
                pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
        """
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, encoder_depth)

        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)

        # initialize other hyper-parameters
        self.corruption_len = int(corruption_rate * input_dim)
        self.contrastive_loss_temperature = contrastive_loss_temperature
        if not model_device:
            print('Model requires a device to run on!')
            exit(1)
        self.device = model_device
        print(f"Created a model with input dimension: {input_dim}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, input_batch):
        # compute embeddings
        emb_batch = self.encoder(input_batch)
        emb_batch = self.pretraining_head(emb_batch)

        return emb_batch
    
    def _contrastive_loss(self, z_i, z_j):
        """
        NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        hyper-parameter: temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        
        Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()
        numerator = torch.exp(positives / self.contrastive_loss_temperature)
        denominator = mask * torch.exp(similarity / self.contrastive_loss_temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss

    
    def get_dataset_embedding(self, input, one_hot_encoder):
        self.eval()
        input = one_hot_encoder.transform(input)

        with torch.no_grad():
            input = torch.tensor(input, dtype=torch.float32).to(self.device)
            embedding = self.encoder(input)

        return embedding.numpy()

    
    def train_epoch(self, data_sampler, optimizer, one_hot_encoder):
        self.train()
        epoch_loss = 0.0

        for i in range(data_sampler.n_batches):
            anchors, random_samples = data_sampler.sample_batch()
            # firstly, corrupt on the original pandas dataframe
            corruption_masks = np.zeros_like(anchors, dtype=bool)
            for i in range(np.shape(anchors)[0]):
                corruption_idxes = np.random.permutation(np.shape(anchors)[1])[:self.corruption_len]
                corruption_masks[i, corruption_idxes] = True
            anchors_corrupted = np.where(corruption_masks, random_samples, anchors)

            anchors, anchors_corrupted = one_hot_encoder.transform(pd.DataFrame(data=anchors,columns=data_sampler.columns)), \
                                        one_hot_encoder.transform(pd.DataFrame(data=anchors_corrupted,columns=data_sampler.columns))

            anchors, anchors_corrupted = torch.tensor(anchors, dtype=torch.float32).to(self.device), \
                                            torch.tensor(anchors_corrupted, dtype=torch.float32).to(self.device)

            # reset gradients
            optimizer.zero_grad()

            # get embeddings
            emb_anchors = self(anchors)
            emb_corrupted = self(anchors_corrupted)

            # compute loss
            loss = self._contrastive_loss(emb_anchors, emb_corrupted)
            loss.backward()

            # update model weights
            optimizer.step()

            # log progress
            epoch_loss += anchors.size(0) * loss.item()

        return epoch_loss / len(data_sampler)
