import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_contrastive_loss(model, data_sampler, mask_generator, optimizer, DEVICE, n_epochs_max, n_epochs_min, early_stopping):
    early_stopping_patience = 3
    train_losses, valid_losses = [], []

    for i in tqdm(range(n_epochs_max)):
        #<<<<<<TRAIN>>>>>"
        model.train()
        epoch_loss = 0
        for j in range(data_sampler['train'].n_batches):
            anchors, random_samples = data_sampler['train'].sample_batch()
            # firstly, corrupt on the original pandas dataframe
            corruption_masks = mask_generator.get_masks(np.shape(anchors)[0])
            assert np.shape(anchors) == np.shape(corruption_masks)
            anchors_corrupted = np.where(corruption_masks, random_samples, anchors)

            anchors, anchors_corrupted = torch.tensor(anchors, dtype=torch.float32).to(DEVICE), \
                                            torch.tensor(anchors_corrupted, dtype=torch.float32).to(DEVICE)

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
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / data_sampler['train'].n_batches)

        #<<<<<<Validation>>>>>
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for j in range(data_sampler['valid'].n_batches):
                anchors, random_samples = data_sampler['valid'].sample_batch()
                # firstly, corrupt on the original pandas dataframe
                corruption_masks = mask_generator.get_masks(np.shape(anchors)[0])
                assert np.shape(anchors) == np.shape(corruption_masks)

                anchors_corrupted = np.where(corruption_masks, random_samples, anchors)

                anchors, anchors_corrupted = torch.tensor(anchors, dtype=torch.float32).to(DEVICE), \
                                                torch.tensor(anchors_corrupted, dtype=torch.float32).to(DEVICE)

                # get embeddings
                emb_final_anchors = model.get_final_embedding(anchors)
                emb_final_corrupted = model.get_final_embedding(anchors_corrupted)

                # compute loss
                loss = model.contrastive_loss(emb_final_anchors, emb_final_corrupted)

                # log progress
                epoch_loss += loss.item()

        valid_losses.append(epoch_loss / data_sampler['valid'].n_batches)

        if early_stopping:
            # Decide if stopping already
            if (i+1) > max(early_stopping_patience, n_epochs_min) and \
                valid_losses[-early_stopping_patience-1] >= np.max(valid_losses[-early_stopping_patience:]) :
                print(f"Reached early stopping with patience {early_stopping_patience}. Stopping after epoch {i+1} (>{n_epochs_min})!")
                break

    assert len(train_losses) == len(valid_losses) == i+1

    return train_losses, valid_losses

def train_classification(model, data_sampler, optimizer, DEVICE, n_epochs_max, n_epochs_min, early_stopping):
    early_stopping_patience = 3
    train_losses, valid_losses = [], []

    for i in tqdm(range(n_epochs_max)):
        # <<<<<<<<<TRAIN>>>>>>>>>
        model.train()
        epoch_loss = 0.0
        for j in range(data_sampler['train'].n_batches):
            inputs, targets = data_sampler['train'].sample_batch()

            inputs = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)
            # seemingly int64 is often used as the type for indices
            targets = torch.tensor(targets, dtype=torch.int64).to(DEVICE)

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

        train_losses.append(epoch_loss / data_sampler['train'].n_batches)

        #<<<<<<<<Validation>>>>>>>>>>
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for j in range(data_sampler['valid'].n_batches):
                inputs, targets = data_sampler['valid'].sample_batch()

                inputs = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)
                # seemingly int64 is often used as the type for indices
                targets = torch.tensor(targets, dtype=torch.int64).to(DEVICE)
                    
                # get classification predictions
                pred_logits = model.get_classification_prediction_logits(inputs)

                # compute loss
                loss = model.classification_loss(pred_logits, targets)

                # log progress
                epoch_loss += loss.item()

        valid_losses.append(epoch_loss / data_sampler['valid'].n_batches)

        if early_stopping:
            # Decide if stopping already
            if (i+1) > max(early_stopping_patience, n_epochs_min) and \
                valid_losses[-early_stopping_patience-1] >= np.max(valid_losses[-early_stopping_patience:]) :
                print(f"Reached early stopping with patience {early_stopping_patience}. Stopping after epoch {i+1} (>{n_epochs_min})!")
                break

    assert len(train_losses) == len(valid_losses) == i+1

    return train_losses, valid_losses
 