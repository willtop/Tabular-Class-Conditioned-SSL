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
        encoder=None,
        pretraining_head=None,
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

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = MLP(input_dim, emb_dim, encoder_depth)

        if pretraining_head:
            self.pretraining_head = pretraining_head
        else:
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_sample):
        batch_size, m = anchor.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted

        corruption_mask = torch.zeros_like(anchor, dtype=torch.bool)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        positive = torch.where(corruption_mask, random_sample, anchor)

        # compute embeddings
        emb_anchor = self.encoder(anchor)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_positive = self.encoder(positive)
        emb_positive = self.pretraining_head(emb_positive)

        return emb_anchor, emb_positive
    
    def contrastive_loss(self, z_i, z_j):
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

    
    def get_dataset_embeddings(self, loader):
        self.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in loader:
                anchor = anchor.to(self.device)
                embeddings.append(self.encoder(anchor))

        embeddings = torch.cat(embeddings).numpy()

        return embeddings
    
    def train_epoch(self, train_loader, optimizer):
        self.train()
        epoch_loss = 0.0

        for anchor, positive in train_loader:
            anchor, positive = anchor.to(self.device), positive.to(self.device)

            # reset gradients
            optimizer.zero_grad()

            # get embeddings
            emb_anchor, emb_positive = self(anchor, positive)

            # compute loss
            loss = self.contrastive_loss(emb_anchor, emb_positive)
            loss.backward()

            # update model weights
            optimizer.step()

            # log progress
            epoch_loss += anchor.size(0) * loss.item()

        return epoch_loss / len(train_loader.dataset)
