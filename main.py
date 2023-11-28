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
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm

from model import Neural_Net
from dataset_samplers import RandomCorruptSampler, ClassCorruptSampler, SupervisedSampler 
from corruption_mask_generators import RandomMaskGenerator, CrossClusterMaskGenerator
from training import train_contrastive_loss, train_classification
from utils import fix_seed, load_openml_list, preprocess_datasets, get_bootstrapped_targets

import warnings
warnings.filterwarnings('ignore')
print("Disabled warnings!")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")

FREEZE_PRETRAINED_ENCODER = True
CONTRASTIVE_LEARNING_MAX_EPOCHS = 500
SUPERVISED_LEARNING_MAX_EPOCHS = 200

FRACTION_LABELED = 0.15
CORRUPTION_RATE = 0.5
BATCH_SIZE = 128
DIDS = [11, 54, 1050]
SEEDS = [614579, 336466, 974761, 450967]#, 743562, 767734]
CORRUPT_METHODS = ['rand_corr', 'cls_corr', 'orc_corr', 'cluster_corr']
CORRUPT_LOCATIONS = ['rand_feats', 'crossCluster_feats']
ALL_METHODS = ['no_pretrain'] + [f'{i}-{j}' for i in CORRUPT_METHODS for j in CORRUPT_LOCATIONS]

if __name__ == "__main__":    
    res_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                                'experiments', \
                                "accuracies.txt")
    os.makedirs(os.path.dirname(res_file), exist_ok=True) 
    # clear the file
    print(f"Preparing and clearing file {res_file} for writing results...")
    with open(res_file, "w") as res_f:
        res_f.write(f"Experiment specs: Corruption rate: {CORRUPTION_RATE}; Fraction of data labeled: {FRACTION_LABELED};" +  
                    f"Number of seeds: {len(SEEDS)};" + 
                    f"contrastive learning max epochs: {CONTRASTIVE_LEARNING_MAX_EPOCHS};" + 
                    f"supervised learning max epochs: {SUPERVISED_LEARNING_MAX_EPOCHS};" + 
                    f"whether freeze pretrained encoders: {FREEZE_PRETRAINED_ENCODER}.\n")

    # OpenML dataset
    all_datasets = load_openml_list(DIDS)

    for ds in all_datasets:
        dataset_name, data, target = ds
        print(f"Loaded dataset: {dataset_name}, with data shape: {data.shape}, and target shape: {target.shape}")
        assert len(data.select_dtypes(include='category').columns)==0
        accuracies = {}
        for key in ALL_METHODS:
            accuracies[key] = []       

        # run each experiment multiple times with varying seeds
        for seed in SEEDS:
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
            train_data, valid_data, test_data = preprocess_datasets( \
                    train_data, valid_data, test_data, normalize_numerical_features=True)
            label_encoder_target = preprocessing.LabelEncoder()
            train_targets = label_encoder_target.fit_transform(train_targets)
            valid_targets = label_encoder_target.transform(valid_targets)
            test_targets = label_encoder_target.transform(test_targets)
            
            # separate out labeled subset
            n_train_samples_labeled = int(len(train_data)*FRACTION_LABELED)
            idxes_tmp = np.random.permutation(len(train_data))[:n_train_samples_labeled]
            mask_train_labeled = np.zeros(len(train_data), dtype=bool)
            mask_train_labeled[idxes_tmp] = True
            n_valid_samples_labeled = int(len(valid_data)*FRACTION_LABELED)
            idxes_tmp = np.random.permutation(len(valid_data))[:n_valid_samples_labeled]
            mask_valid_labeled = np.zeros(len(valid_data), dtype=bool)
            mask_valid_labeled[idxes_tmp] = True
            supervised_sampler = {}
            supervised_sampler['train'] = SupervisedSampler(data=train_data[mask_train_labeled], BATCH_SIZE=BATCH_SIZE, target=train_targets[mask_train_labeled])
            supervised_sampler['valid'] = SupervisedSampler(data=valid_data[mask_valid_labeled], BATCH_SIZE=BATCH_SIZE, target=valid_targets[mask_valid_labeled])

            # prepare models
            models, contrastive_loss_histories, supervised_loss_histories, contrastive_optimizers, supervised_optimizers = {}, {}, {}, {}, {}
            for key in ALL_METHODS:
                models[key] = Neural_Net(
                    input_dim=train_data.shape[1],  # model expect one-hot encoded input
                    emb_dim=256,
                    output_dim=n_classes,
                    model_DEVICE=DEVICE    
                ).to(DEVICE)
                contrastive_loss_histories[key] = {'train': [], 'valid': []}
                supervised_loss_histories[key] = {'train': [], 'valid': []}

            # Firstly, train the supervised learning model on the labeled subset
            supervised_optimizers['no_pretrain'] = Adam(models['no_pretrain'].parameters(), lr=0.001)    
            print(f"Supervised learning for no_pretrain...")
            train_losses, valid_losses = train_classification(models['no_pretrain'], 
                                                              supervised_sampler, 
                                                              supervised_optimizers['no_pretrain'], 
                                                              DEVICE, 
                                                              n_epochs_max=SUPERVISED_LEARNING_MAX_EPOCHS,
                                                              n_epochs_min=50, 
                                                              early_stopping=False)
            supervised_loss_histories['no_pretrain']['train'] = train_losses
            supervised_loss_histories['no_pretrain']['valid'] = valid_losses

            ############# Prepare data samplers for corruption ############
            # Use supervised model to obtain pseudo labels
            contrastive_samplers = {}
            for key in CORRUPT_METHODS:
                contrastive_samplers[key] = {}
            # Random Sampling: Not using class information in original corruption
            contrastive_samplers['rand_corr']['train'] = RandomCorruptSampler(train_data, BATCH_SIZE) 
            contrastive_samplers['rand_corr']['valid'] = RandomCorruptSampler(valid_data, BATCH_SIZE) 
            # Oracle Class Sampling: Using oracle info on training labels
            contrastive_samplers['orc_corr']['train'] = ClassCorruptSampler(train_data, BATCH_SIZE, train_targets) 
            contrastive_samplers['orc_corr']['valid'] = ClassCorruptSampler(valid_data, BATCH_SIZE, valid_targets)
            # Predicted Class Sampling
            bootstrapped_train_targets = get_bootstrapped_targets( \
                train_data, train_targets, models['no_pretrain'], mask_train_labeled, DEVICE)
            contrastive_samplers['cls_corr']['train'] = ClassCorruptSampler(train_data, BATCH_SIZE, bootstrapped_train_targets) 
            bootstrapped_valid_targets = get_bootstrapped_targets( \
                valid_data, valid_targets, models['no_pretrain'], mask_valid_labeled, DEVICE)
            contrastive_samplers['cls_corr']['valid'] = ClassCorruptSampler(valid_data, BATCH_SIZE, bootstrapped_valid_targets)
            # Unsupervised Cluster Based Sampling
            pca_10D = PCA(n_components=min(train_data.shape[1], 10), copy=True)
            train_data_tmp = pca_10D.fit_transform(train_data)
            valid_data_tmp = pca_10D.transform(valid_data)
            kmeans = KMeans(n_clusters=n_classes)
            train_cluster_assignments = kmeans.fit_predict(train_data_tmp)
            valid_cluster_assignments = kmeans.predict(valid_data_tmp)
            contrastive_samplers['cluster_corr']['train'] = ClassCorruptSampler(train_data, BATCH_SIZE, train_cluster_assignments)
            contrastive_samplers['cluster_corr']['valid'] = ClassCorruptSampler(valid_data, BATCH_SIZE, valid_cluster_assignments)

            ################ Prepare feature selections for masking #############
            # prepare mask generator
            mask_generators = {}
            mask_generators['rand_feats'] = RandomMaskGenerator(train_data.shape[1], CORRUPTION_RATE)
            mask_generators['crossCluster_feats'] = CrossClusterMaskGenerator(train_data.shape[1], CORRUPTION_RATE)
            mask_generators['crossCluster_feats'].fit_feature_clusters(train_data)

            ################ Contrastive training #############
            for corrupt_method in CORRUPT_METHODS:
                for corrupt_loc in CORRUPT_LOCATIONS:
                    method_key = f"{corrupt_method}-{corrupt_loc}"
                    contrastive_optimizers[method_key] = Adam(models[method_key].parameters(), lr=0.001)
                    print(f"Contrastive learning for {method_key}....")
                    train_losses, valid_losses = train_contrastive_loss(models[method_key], 
                                                                        contrastive_samplers[corrupt_method], 
                                                                        mask_generators[corrupt_loc],
                                                                        contrastive_optimizers[method_key], 
                                                                        DEVICE, 
                                                                        n_epochs_max=CONTRASTIVE_LEARNING_MAX_EPOCHS,
                                                                        n_epochs_min=100, 
                                                                        early_stopping=False)
                    contrastive_loss_histories[method_key]['train'] = train_losses
                    contrastive_loss_histories[method_key]['valid'] = valid_losses

            # fine tune the pre-trained models on the down-stream supervised learning task
            for corrupt_method in CORRUPT_METHODS:
                for corrupt_loc in CORRUPT_LOCATIONS:
                    method_key = f"{corrupt_method}-{corrupt_loc}"
                    if FREEZE_PRETRAINED_ENCODER:
                        models[method_key].freeze_encoder()
                    supervised_optimizers[method_key] = Adam(filter(lambda p: p.requires_grad, models[method_key].parameters()), lr=0.001)
                    supervised_loss_histories[method_key] = {'train': [], 'valid': []}

            for corrupt_method in CORRUPT_METHODS:
                for corrupt_loc in CORRUPT_LOCATIONS:
                    method_key = f"{corrupt_method}-{corrupt_loc}"
                    print(f"Supervised fine-tuning for {method_key}...")
                    train_losses, valid_losses = train_classification(models[method_key], 
                                                                      supervised_sampler, 
                                                                      supervised_optimizers[method_key], 
                                                                      DEVICE, 
                                                                      n_epochs_max=SUPERVISED_LEARNING_MAX_EPOCHS,
                                                                      n_epochs_min=50, 
                                                                      early_stopping=False)
                    supervised_loss_histories[method_key]['train'] = train_losses
                    supervised_loss_histories[method_key]['valid'] = valid_losses

            # Evaluation on prediction accuracies
            for method_key in ALL_METHODS:
                models[method_key].eval()
                with torch.no_grad():
                    test_prediction_logits = models[method_key].get_classification_prediction_logits( \
                                        torch.tensor(test_data, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                    test_predictions = np.argmax(test_prediction_logits,axis=1)
                    accuracies[method_key].append(np.mean(test_predictions==test_targets)*100)

        # write the results to a file        
        with open(res_file, 'a') as res_f:
            res_f.write(f"Dataset: {dataset_name}\n")
            for method_key in ALL_METHODS:
                avg_accuracy = np.mean(accuracies[method_key])
                accuracy_std = np.std(accuracies[method_key])
                res_f.write(f"{method_key} accuracy avg {avg_accuracy:.3f}; std {accuracy_std:.3f} | ")
            res_f.write("\n")
        
        print(f"{dataset_name} finished!")

    print("Main function finished!") 
