import random
import numpy as np
import pandas as pd
import torch
import openml
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_openml_classification(did):
    dataset = openml.datasets.get_dataset(did)
    # since under SCARF corruption, the replacement by sampling happens before one-hot encoding, load the 
    # data in its original form
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)        

    order = np.arange(y.shape[0])
    # Don't think need to re-seed here
    np.random.shuffle(order)
    X, y = X.iloc[order], y.iloc[order]

    # No need to keep categorical indicators and attribute names, as they can be readily obtained from pandas dataframe
    print(f"Dataset with did {did} has {sum(categorical_indicator)}/{len(attribute_names)} categorical features.")
    return X, y

def load_openml_list(dids):
    datasets = []
    datasets_list = openml.datasets.list_datasets(dids, output_format='dataframe')

    for ds in datasets_list.index:
        entry = datasets_list.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported for now")
            exit(1)
        else:
            X, y = get_openml_classification(int(entry.did))
        if X is None:
            continue

        datasets += [[entry['name'], X, y]]

    return datasets

def preprocess_datasets(train_data, valid_data, test_data, normalize_numerical_features):
    assert isinstance(train_data, pd.DataFrame) and \
            isinstance(valid_data, pd.DataFrame) and \
             isinstance(test_data, pd.DataFrame)
    assert np.all(train_data.columns == valid_data.columns) and \
            np.all(valid_data.columns == test_data.columns)
    features_dropped = []
    for col in train_data.columns:
        # drop columns with all null values or with a constant value on training data
        if train_data[col].isnull().all() or train_data[col].nunique() == 1:
            train_data.drop(columns=col, inplace=True)
            valid_data.drop(columns=col, inplace=True)
            test_data.drop(columns=col, inplace=True)
            features_dropped.append(col)
            continue
        # fill the missing values
        if train_data[col].isnull().any() or \
            valid_data[col].isnull().any() or \
             test_data[col].isnull().any():
            # for categorical features, fill with the mode in the training data
            if train_data[col].dtype.name == 'category':
                val_fill = train_data[col].mode(dropna=True)[0]
                train_data[col].fillna(val_fill, inplace=True)
                valid_data[col].fillna(val_fill, inplace=True)
                test_data[col].fillna(val_fill, inplace=True)
            # for numerical features, fill with the mean of the training data
            
            else:
                val_fill = train_data[col].mean(skipna=True)
                train_data[col].fillna(val_fill, inplace=True)
                valid_data[col].fillna(val_fill, inplace=True)
                test_data[col].fillna(val_fill, inplace=True)

    if normalize_numerical_features:
        # z-score transform numerical values
        scaler = StandardScaler()
        non_categorical_columns = train_data.select_dtypes(exclude='category').columns
        if len(non_categorical_columns) == 0:
            print("No numerical features present! Skip numerical z-score normalization.")
        else:
            train_data[non_categorical_columns] = scaler.fit_transform(train_data[non_categorical_columns])
            valid_data[non_categorical_columns] = scaler.transform(valid_data[non_categorical_columns])
            test_data[non_categorical_columns] = scaler.transform(test_data[non_categorical_columns])

    print(f"Data preprocess finished! Dropped {len(features_dropped)} features: {features_dropped}. {'Normalized numerical features.' if normalize_numerical_features else ''}")
    return

def fit_one_hot_encoder(one_hot_encoder_raw, train_data):
    categorical_columns = train_data.select_dtypes(include='category').columns
    one_hot_encoder = make_column_transformer((one_hot_encoder_raw, categorical_columns), remainder='passthrough')
    one_hot_encoder.fit(train_data)
    return one_hot_encoder
