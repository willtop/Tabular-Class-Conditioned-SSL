import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import torch
from sklearn import datasets, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm

from models.model import Neural_Net
from dataset_samplers import RandomCorruptSampler, ClassCorruptSampler, SupervisedSampler 
from training import train_contrastive_loss, train_classification
from utils import fix_seed, load_openml_list, preprocess_datasets, fit_one_hot_encoder, get_bootstrapped_targets

import warnings
warnings.filterwarnings('ignore')
print("Disabled warnings!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fraction_withLabel = 0.25
    batch_size = 128

    res_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                                'experiments', \
                                "accuracies.txt")
    os.makedirs(os.path.dirname(res_file), exist_ok=True) 
    # clear the file
    print(f"Preparing and clearing file {res_file} for writing results...")
    open(res_file, "w").close()

    dids = [23, 4538, 6332, 40975]
    seeds = [614579, 336466, 974761, 450967, 743562, 767734]
    # OpenML dataset
    datasets = load_openml_list([23, 4538, 6332, 40975])

    for i in range(len(dids)):
        dataset = datasets[i]
        dataset_name, data, target = dataset
        print(f"Loaded dataset: {dataset_name}, with data shape: {data.shape}, and target shape: {target.shape}")
        print(f"{dataset_name} dataset has {len(data.select_dtypes(include='category').columns)}/{len(data.columns)} categorical features.")
        accuracies = {}
        for key in ['no_pretrain', 'rand_corr', 'cls_corr', 'orc_corr']:
            accuracies[key] = []       

        # run each experiment multiple times with varying seeds
        for seed in seeds:
            fix_seed(seed)
            # train, validation, test splits
            tmp_data, test_data, tmp_target, test_targets = train_test_split(
                data, target, test_size=0.2, stratify=target, random_state=seed)
            train_data, valid_data, train_targets, valid_targets = train_test_split(
                tmp_data, tmp_target, test_size=1/8, stratify=tmp_target, random_state=seed)
            assert np.all(np.unique(train_targets).sort() == np.unique(valid_targets).sort()) and \
                    np.all(np.unique(train_targets).sort() == np.unique(test_targets).sort())
            n_classes = len(np.unique(train_targets))

            # preprocess datasets
            preprocess_datasets(train_data, valid_data, test_data, normalize_numerical_features=True)
            one_hot_encoder = fit_one_hot_encoder(preprocessing.OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False), train_data)
            label_encoder_target = preprocessing.LabelEncoder()
            train_targets = label_encoder_target.fit_transform(train_targets)
            valid_targets = label_encoder_target.transform(valid_targets)
            test_targets = label_encoder_target.transform(test_targets)
            
            # separate out labeled subset
            n_train_samples_labeled = int(len(train_data)*fraction_withLabel)
            idxes_tmp = np.random.permutation(len(train_data))[:n_train_samples_labeled]
            mask_train_labeled = np.zeros(len(train_data), dtype=bool)
            mask_train_labeled[idxes_tmp] = True
            n_valid_samples_labeled = int(len(valid_data)*fraction_withLabel)
            idxes_tmp = np.random.permutation(len(valid_data))[:n_valid_samples_labeled]
            mask_valid_labeled = np.zeros(len(valid_data), dtype=bool)
            mask_valid_labeled[idxes_tmp] = True
            supervised_sampler = {}
            supervised_sampler['train'] = SupervisedSampler(data=train_data[mask_train_labeled], batch_size=batch_size, target=train_targets[mask_train_labeled])
            supervised_sampler['valid'] = SupervisedSampler(data=valid_data[mask_valid_labeled], batch_size=batch_size, target=valid_targets[mask_valid_labeled])

            # prepare models
            models, contrastive_loss_histories, supervised_loss_histories = {}, {}, {}
            for key in ['no_pretrain', 'rand_corr', 'cls_corr', 'orc_corr']:
                models[key] = Neural_Net(
                    input_dim=one_hot_encoder.transform(train_data).shape[1],  # model expect one-hot encoded input
                    emb_dim=256,
                    output_dim=n_classes,
                    model_device=device    
                ).to(device)
                contrastive_loss_histories[key] = {'train': [], 'valid': []}
                supervised_loss_histories[key] = {'train': [], 'valid': []}
            contrastive_optimizers, supervised_optimizers = {}, {}

            # Firstly, train the supervised learning model on the labeled subset
            supervised_optimizers['no_pretrain'] = Adam(models['no_pretrain'].parameters(), lr=0.001)    
            print(f"Supervised learning for no_pretrain...")
            train_losses, valid_losses = train_classification(      \
                    models['no_pretrain'], supervised_sampler, supervised_optimizers['no_pretrain'], one_hot_encoder, device, min_epochs=50, early_stopping=False)
            supervised_loss_histories['no_pretrain']['train'] = train_losses
            supervised_loss_histories['no_pretrain']['valid'] = valid_losses

            # Use supervised model to obtain pseudo labels
            contrastive_samplers = {}
            for key in ['rand_corr', 'orc_corr', 'cls_corr']:
                contrastive_samplers[key] = {}
            # Random Sampling: Not using class information in original corruption
            contrastive_samplers['rand_corr']['train'] = RandomCorruptSampler(train_data, batch_size) 
            contrastive_samplers['rand_corr']['valid'] = RandomCorruptSampler(valid_data, batch_size) 
            # Oracle Class Sampling: Using oracle info on training labels
            contrastive_samplers['orc_corr']['train'] = ClassCorruptSampler(train_data, batch_size, train_targets) 
            contrastive_samplers['orc_corr']['valid'] = ClassCorruptSampler(valid_data, batch_size, valid_targets)
            # Predicted Class Sampling
            bootstrapped_train_targets = get_bootstrapped_targets( \
                train_data, train_targets, models['no_pretrain'], mask_train_labeled, one_hot_encoder, device)
            contrastive_samplers['cls_corr']['train'] = ClassCorruptSampler(train_data, batch_size, bootstrapped_train_targets) 
            bootstrapped_valid_targets = get_bootstrapped_targets( \
                valid_data, valid_targets, models['no_pretrain'], mask_valid_labeled, one_hot_encoder, device)
            contrastive_samplers['cls_corr']['valid'] = ClassCorruptSampler(valid_data, batch_size, bootstrapped_valid_targets)

            # Contrastive training
            for key in ['rand_corr', 'cls_corr', 'orc_corr']:
                contrastive_optimizers[key] = Adam(models[key].parameters(), lr=0.001)
                print(f"Contrastive learning for {key} sampling....")
                train_losses, valid_losses = train_contrastive_loss(models[key], contrastive_samplers[key], contrastive_optimizers[key], one_hot_encoder, device, min_epochs=100, early_stopping=False)
                contrastive_loss_histories[key]['train'] = train_losses
                contrastive_loss_histories[key]['valid'] = valid_losses

            # fine tune the pre-trained models on the down-stream supervised learning task
            for key in ['rand_corr', 'cls_corr', 'orc_corr']:
                models[key].freeze_encoder()
                supervised_optimizers[key] = Adam(filter(lambda p: p.requires_grad, models[key].parameters()), lr=0.001)
                supervised_loss_histories[key] = {'train': [], 'valid': []}

            for key in ['rand_corr', 'cls_corr', 'orc_corr']:
                print(f"Supervised learning for {key}...")
                train_losses, valid_losses = train_classification(models[key], supervised_sampler, supervised_optimizers[key], one_hot_encoder, device, min_epochs=50, early_stopping=False)
                supervised_loss_histories[key]['train'] = train_losses
                supervised_loss_histories[key]['valid'] = valid_losses

            # Evaluation on prediction accuracies
            for key in ['no_pretrain', 'rand_corr', 'cls_corr', 'orc_corr']:
                models[key].eval()
                with torch.no_grad():
                    test_prediction_logits = models[key].get_classification_prediction_logits(torch.tensor(one_hot_encoder.transform(test_data), dtype=torch.float32).to(device)).cpu().numpy()
                    test_predictions = np.argmax(test_prediction_logits,axis=1)
                    accuracies[key].append(np.mean(test_predictions==test_targets)*100)

        # write the results to a file        
        with open(res_file, 'a') as res_f:
            res_f.write(f"Dataset: {dataset_name}\n")
            for key in ['no_pretrain', 'rand_corr', 'cls_corr', 'orc_corr']:
                avg_accuracy = np.mean(accuracies[key])
                res_f.write(f"{key} {avg_accuracy:.3f}  ")
            res_f.write("\n")
        
        print(f"{dataset_name} finished!")

    print("Main function finished!") 
