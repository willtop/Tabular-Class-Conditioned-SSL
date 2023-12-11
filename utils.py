import random
import numpy as np
import pandas as pd
import torch
import openml
from sklearn.preprocessing import StandardScaler


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_openml_list(DIDS):
    datasets = []
    datasets_list = openml.datasets.list_datasets(DIDS, output_format='dataframe')

    for ds in datasets_list.index:
        entry = datasets_list.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported for now")
            exit(1)    
        else:
            dataset = openml.datasets.get_dataset(int(entry.did))
            # since under SCARF corruption, the replacement by sampling happens before one-hot encoding, load the 
            # data in its original form
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            if np.any(categorical_indicator):
                print(f"Dataset {entry['name']} with did {int(entry.did)} has at least one categorical feature, skipping...")
                continue
            assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)        

            order = np.arange(y.shape[0])
            # Don't think need to re-seed here
            np.random.shuffle(order)
            X, y = X.iloc[order], y.iloc[order]

            assert X is not None

        datasets += [[entry['name'], entry.did, X, y]]

    return datasets

def preprocess_datasets(train_data, test_data, normalize_numerical_features):
    assert isinstance(train_data, pd.DataFrame) and \
             isinstance(test_data, pd.DataFrame)
    assert np.all(train_data.columns == test_data.columns)
    features_dropped = []
    for col in train_data.columns:
        # drop columns with all null values or with a constant value on training data
        if train_data[col].isnull().all() or train_data[col].nunique() == 1:
            train_data.drop(columns=col, inplace=True)
            test_data.drop(columns=col, inplace=True)
            features_dropped.append(col)
            continue
        # fill the missing values
        if train_data[col].isnull().any() or test_data[col].isnull().any():
            # for numerical features, fill with the mean of the training data
            val_fill = train_data[col].mean(skipna=True)
            train_data[col].fillna(val_fill, inplace=True)
            test_data[col].fillna(val_fill, inplace=True)

    if normalize_numerical_features:
        # z-score transform numerical values
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

    print(f"Data preprocess finished! Dropped {len(features_dropped)} features: {features_dropped}. {'Normalized numerical features.' if normalize_numerical_features else ''}")

    # Since all numerical features, convert them into numpy array here
    return np.array(train_data), np.array(test_data)

def get_bootstrapped_targets(data, targets, classifier_model, mask_labeled, DEVICE):
    # use the classifier to predict for all data first
    classifier_model.eval()
    with torch.no_grad():
        pred_logits = classifier_model.get_classification_prediction_logits(torch.tensor(data,dtype=torch.float32).to(DEVICE)).cpu().numpy()
    preds = np.argmax(pred_logits, axis=1)
    return np.where(mask_labeled, targets, preds)
