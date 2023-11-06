import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
        emb_final_anchors = model.get_final_embedding(anchors)
        emb_final_corrupted = model.get_final_embedding(anchors_corrupted)

        # compute loss
        loss = model.contrastive_loss(emb_final_anchors, emb_final_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += anchors.size(0) * loss.item()

    return epoch_loss / len(data_sampler)

def train_classification_epoch(model, data_sampler, optimizer, one_hot_encoder, device):
    model.train()
    epoch_loss = 0.0

    for i in range(data_sampler.n_batches):
        inputs, targets = data_sampler.sample_batch()
        

        inputs = one_hot_encoder.transform(pd.DataFrame(data=inputs,columns=data_sampler.columns))

        inputs = torch.tensor(inputs.astype(float), dtype=torch.float32).to(device)
        # seemingly int64 is often used as the type for indices
        targets = torch.tensor(targets.astype(float), dtype=torch.int64).to(device)

        # reset gradients
        optimizer.zero_grad()

        # get classification predictions
        pred_logits = model.get_classification_prediction_logits(inputs)

        # compute loss
        loss = model.classification_loss(pred_logits, targets)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += loss.item()

    return epoch_loss / data_sampler.n_batches
 