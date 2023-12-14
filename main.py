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
from corruption_mask_generators import RandomMaskGenerator, CorrelationMaskGenerator
from training import train_contrastive_loss, train_classification
from utils import fix_seed, load_openml_list, preprocess_datasets, get_bootstrapped_targets, compute_feature_mutual_influences

import warnings
warnings.filterwarnings('ignore')
print("Disabled warnings!")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")

CONTRASTIVE_LEARNING_MAX_EPOCHS = 500
SUPERVISED_LEARNING_MAX_EPOCHS = 100

FRACTION_LABELED = 0.1
CORRUPTION_RATE = 0.3
BATCH_SIZE = 256
ALL_DIDS = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1510, 1494, 1480, 1462, 1464, 6332, 23381, 40966, 40982, 40994, 40975]
SEEDS = [614579, 336466, 974761, 450967, 743562]
CORRUPT_METHODS = ['rand_corr', 'cls_corr', 'orc_corr']
CORRUPT_LOCATIONS = ['rand_feats', 'leastCorr_feats', 'mostCorr_feats']
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
                    f"contrastive learning epochs: {CONTRASTIVE_LEARNING_MAX_EPOCHS};" + 
                    f"supervised learning epochs: {SUPERVISED_LEARNING_MAX_EPOCHS}.\n")

    # OpenML dataset
    all_datasets = load_openml_list(ALL_DIDS)

    for ds in all_datasets:
        dataset_name, dataset_did, data, target = ds
        print(f"Loaded dataset: {dataset_name} ({dataset_did}), with data shape: {data.shape}, and target shape: {target.shape}")
        assert len(data.select_dtypes(include='category').columns)==0
        accuracies = {}
        for key in ALL_METHODS:
            accuracies[key] = []       

        # run each experiment multiple times with varying seeds
        for seed in SEEDS:
            fix_seed(seed)
            # train, test splits
            train_data, test_data, train_targets, test_targets = train_test_split(
                data, target, test_size=0.2, stratify=target, random_state=seed)
            assert np.all(np.unique(train_targets).sort() == np.unique(test_targets).sort())
            n_classes = len(np.unique(train_targets))

            # preprocess datasets
            train_data, test_data = preprocess_datasets(train_data, test_data, normalize_numerical_features=True)
            label_encoder_target = preprocessing.LabelEncoder()
            train_targets = label_encoder_target.fit_transform(train_targets)
            test_targets = label_encoder_target.transform(test_targets)
            print(f"Class distributions:")
            unique, counts = np.unique(train_targets, return_counts=True)
            print(f"Training: cls: {unique}; counts: {counts}")
            unique, counts = np.unique(test_targets, return_counts=True)
            print(f"Testing: cls: {unique}; counts: {counts}")
            
            # separate out labeled subset
            n_train_samples_labeled = int(len(train_data)*FRACTION_LABELED)
            idxes_tmp = np.random.permutation(len(train_data))[:n_train_samples_labeled]
            mask_train_labeled = np.zeros(len(train_data), dtype=bool)
            mask_train_labeled[idxes_tmp] = True
            supervised_sampler = SupervisedSampler(data=train_data[mask_train_labeled], batch_size=BATCH_SIZE, target=train_targets[mask_train_labeled])

            # train xgboost to learn predicting one feature based on the rest
            # use learned feature importance to identify feature correlations
            feat_impt, feat_impt_range = compute_feature_mutual_influences(train_data)

            # prepare models
            models, contrastive_loss_histories, supervised_loss_histories, contrastive_optimizers, supervised_optimizers = {}, {}, {}, {}, {}
            for key in ALL_METHODS:
                models[key] = Neural_Net(
                    input_dim=train_data.shape[1],  # model expect one-hot encoded input
                    emb_dim=256,
                    output_dim=n_classes,
                    model_DEVICE=DEVICE    
                ).to(DEVICE)
                contrastive_loss_histories[key] = {}
                supervised_loss_histories[key] = {}

            # Firstly, train the supervised learning model on the labeled subset
            # freeze the supervised learning encoder as initialized
            models['no_pretrain'].freeze_encoder()
            supervised_optimizers['no_pretrain'] = Adam(filter(lambda p: p.requires_grad, models['no_pretrain'].parameters()), lr=0.001)    
            print(f"Supervised learning for no_pretrain...")
            train_losses = train_classification(models['no_pretrain'], 
                                                supervised_sampler, 
                                                supervised_optimizers['no_pretrain'], 
                                                DEVICE, 
                                                n_epochs_max=SUPERVISED_LEARNING_MAX_EPOCHS)
            supervised_loss_histories['no_pretrain'] = train_losses

            ############# Prepare data samplers for corruption ############
            # Use supervised model to obtain pseudo labels
            contrastive_samplers = {}
            for key in CORRUPT_METHODS:
                contrastive_samplers[key] = {}
            # Random Sampling: Not using class information in original corruption
            contrastive_samplers['rand_corr'] = RandomCorruptSampler(train_data, BATCH_SIZE) 
            # Oracle Class Sampling: Using oracle info on training labels
            contrastive_samplers['orc_corr'] = ClassCorruptSampler(train_data, BATCH_SIZE, train_targets) 
            # Predicted Class Sampling
            bootstrapped_train_targets = get_bootstrapped_targets( \
                train_data, train_targets, models['no_pretrain'], mask_train_labeled, DEVICE)
            contrastive_samplers['cls_corr'] = ClassCorruptSampler(train_data, BATCH_SIZE, bootstrapped_train_targets) 

            ################ Prepare feature selections for masking #############
            # prepare mask generator
            mask_generators = {}
            mask_generators['rand_feats'] = RandomMaskGenerator(train_data.shape[1], CORRUPTION_RATE)
            mask_generators['leastCorr_feats'] = CorrelationMaskGenerator(train_data.shape[1], CORRUPTION_RATE, high_correlation=False)
            mask_generators['mostCorr_feats'] = CorrelationMaskGenerator(train_data.shape[1], CORRUPTION_RATE, high_correlation=True)
            mask_generators['leastCorr_feats'].initialize_probabilities(feat_impt)
            mask_generators['mostCorr_feats'].initialize_probabilities(feat_impt)


            ################ Contrastive training #############
            for corrupt_method in CORRUPT_METHODS:
                for corrupt_loc in CORRUPT_LOCATIONS:
                    method_key = f"{corrupt_method}-{corrupt_loc}"
                    contrastive_optimizers[method_key] = Adam(models[method_key].parameters(), lr=0.001)
                    print(f"Contrastive learning for {method_key}....")
                    train_losses = train_contrastive_loss(models[method_key], 
                                                          contrastive_samplers[corrupt_method],
                                                          mask_generators[corrupt_loc],
                                                          contrastive_optimizers[method_key], 
                                                          DEVICE, 
                                                          n_epochs_max=CONTRASTIVE_LEARNING_MAX_EPOCHS)
                    contrastive_loss_histories[method_key] = train_losses

            # fine tune the pre-trained models on the down-stream supervised learning task
            for corrupt_method in CORRUPT_METHODS:
                for corrupt_loc in CORRUPT_LOCATIONS:
                    method_key = f"{corrupt_method}-{corrupt_loc}"
                    models[method_key].freeze_encoder()
                    supervised_optimizers[method_key] = Adam(filter(lambda p: p.requires_grad, models[method_key].parameters()), lr=0.001)
                    supervised_loss_histories[method_key] = {}

            for corrupt_method in CORRUPT_METHODS:
                for corrupt_loc in CORRUPT_LOCATIONS:
                    method_key = f"{corrupt_method}-{corrupt_loc}"
                    print(f"Supervised fine-tuning for {method_key}...")
                    train_losses = train_classification(models[method_key], 
                                                        supervised_sampler, 
                                                        supervised_optimizers[method_key], 
                                                        DEVICE, 
                                                        n_epochs_max=SUPERVISED_LEARNING_MAX_EPOCHS)
                    supervised_loss_histories[method_key] = train_losses

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
            res_f.write(f"Dataset: {dataset_name} ({dataset_did}) with feature importance range {feat_impt_range:.2f}\n")
            for method_key in ALL_METHODS:
                avg_accuracy = np.mean(accuracies[method_key])
                accuracy_std = np.std(accuracies[method_key])
                res_f.write(f"{method_key} accuracy avg {avg_accuracy:.2f}; std {accuracy_std:.2f} |  ")
            res_f.write("\n")
        
        print(f"{dataset_name} finished!")

    print("Main function finished!") 
