import numpy as np
import pandas as pd
import torch

def train_contrastive_loss_epoch(model, data_sampler, optimizer, one_hot_encoder, device):
    model.train()
    epoch_loss = 0.0

    for i in range(data_sampler.n_batches):
        anchors, random_samples = data_sampler.sample_batch()
        # firstly, corrupt on the original pandas dataframe
        corruption_masks = np.zeros_like(anchors, dtype=bool)
        for i in range(np.shape(anchors)[0]):
            corruption_idxes = np.random.permutation(np.shape(anchors)[1])[:model.corruption_len]
            corruption_masks[i, corruption_idxes] = True
        anchors_corrupted = np.where(corruption_masks, random_samples, anchors)

        anchors, anchors_corrupted = one_hot_encoder.transform(pd.DataFrame(data=anchors,columns=data_sampler.columns)), \
                                    one_hot_encoder.transform(pd.DataFrame(data=anchors_corrupted,columns=data_sampler.columns))

        anchors, anchors_corrupted = torch.tensor(anchors.astype(float), dtype=torch.float32).to(device), \
                                        torch.tensor(anchors_corrupted.astype(float), dtype=torch.float32).to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_final_anchors = model(anchors)
        emb_final_corrupted = model(anchors_corrupted)

        # compute loss
        loss = model.contrastive_loss(emb_final_anchors, emb_final_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += anchors.size(0) * loss.item()

    return epoch_loss / len(data_sampler)

def train_classification_epoch(model, data_sampler, optimizer, one_hot_encoder):
    return 0