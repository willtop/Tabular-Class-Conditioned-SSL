import random
import numpy as np
import pandas as pd
import torch
import openml


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

def load_openml_list(dids, filter_for_nan=False):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        datalist = datalist[datalist['NumberOfInstancesWithMissingValues'] == 0]
        print(f'Number of datasets after Nan and feature number filtering: {len(datalist)}')

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
