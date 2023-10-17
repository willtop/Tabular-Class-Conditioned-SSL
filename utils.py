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
    np.random.seed(13)
    np.random.shuffle(order)
    X, y = X.iloc[order], y.iloc[order]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names

def load_openml_list(dids):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist.head()
    for ds in datalist.index:
        entry = datalist.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported")
            #X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            X, y, categorical_feats, attribute_names = get_openml_classification(int(entry.did))
        if X is None:
            continue

        datasets += [[entry['name'], X, y, categorical_feats, attribute_names]]

    return datasets

def preprocess_datasets(train_data, valid_data, test_data):
    assert isinstance(train_data, pd.DataFrame) and \
            isinstance(valid_data, pd.DataFrame) and \
             isinstance(test_data, pd.DataFrame)
    assert np.all(train_data.columns == valid_data.columns) and \
            np.all(valid_data.columns == test_data.columns)
    n_features_dropped = 0
    for col in train_data.columns:
        # drop columns with all null values or with a constant value on training data
        if train_data[col].isnull().all() or train_data[col].nunique() == 1:
            train_data.drop(columns=col, inplace=True)
            valid_data.drop(columns=col, inplace=True)
            test_data.drop(columns=col, inplace=True)
            n_features_dropped += 1
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

    # z-score transform numerical values
    scaler = StandardScaler()
    non_categorical_columns = train_data.select_dtypes(exclude='category').columns
    train_data[non_categorical_columns] = scaler.fit_transform(train_data[non_categorical_columns])
    valid_data[non_categorical_columns] = scaler.transform(valid_data[non_categorical_columns])
    test_data[non_categorical_columns] = scaler.transform(test_data[non_categorical_columns])

    print(f"Data preprocess finished! Dropped {n_features_dropped} features")
    return

def fit_one_hot_encoder(one_hot_encoder_raw, train_data):
    categorical_columns = train_data.select_dtypes(include='category').columns
    one_hot_encoder = make_column_transformer((one_hot_encoder_raw, categorical_columns), remainder='passthrough')
    one_hot_encoder.fit(train_data)
    return one_hot_encoder
