import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset_samplers import ClassCorruptSampler
from utils import *

def _train_contrastive_loss_oneEpoch(model, data_sampler, mask_generator, optimizer):
    model.train()
    epoch_loss = 0
    for _ in range(data_sampler.n_batches):
        anchors, random_samples = data_sampler.sample_batch()
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

    return epoch_loss / data_sampler.n_batches


def train_contrastive_loss(model, method_key, contrastive_sampler, supervised_sampler, mask_generator, mask_train_labeled):
    print(f"Contrastive learning for {method_key}....")
    train_losses = []
    optimizer = initialize_adam_optimizer(model)
    
    for i in tqdm(range(1, CONTRASTIVE_LEARNING_MAX_EPOCHS+1)):
        if i%CLS_CORR_REFRESH_SAMPLER_PERIOD == 0 and 'cls_corr' in method_key:
            model.freeze_encoder()
            # train the current model on down-stream supervised task
            _ = train_classification(model, supervised_sampler)
            # use the current model to do pseudo labeling
            bootstrapped_train_targets = get_bootstrapped_targets( \
                contrastive_sampler.data, contrastive_sampler.target, model, mask_train_labeled)
            # get the class based sampler based on more reliable model predictions
            contrastive_sampler = ClassCorruptSampler(contrastive_sampler.data, bootstrapped_train_targets) 
            model.unfreeze_encoder()
        
        epoch_loss = _train_contrastive_loss_oneEpoch(model, contrastive_sampler, mask_generator, optimizer)
        train_losses.append(epoch_loss)

    return train_losses

def train_classification(model, supervised_sampler):
    train_losses = []
    optimizer = initialize_adam_optimizer(model)
    model.initialize_classification_head()

    for _ in range(SUPERVISED_LEARNING_MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for _ in range(supervised_sampler.n_batches):
            inputs, targets = supervised_sampler.sample_batch()

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

        train_losses.append(epoch_loss / supervised_sampler.n_batches)

    return train_losses
 