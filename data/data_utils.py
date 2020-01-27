import math
import os

import numpy as np
import tensorflow as tf
import pandas as pd

import sys

sys.path.append('..')
import network_utils as utils
 
def random_shuffle(*arrays):
    """Randomly shuffle variable number of np.ndarrays."""
    permutation = np.random.permutation(arrays[0].shape[0])
    shuffled = []
    for array in arrays:
        shuffled.append(array[permutation])
    return tuple(shuffled)

def create_dataset(x, y, batch_size):
    """Create tensorflow dataset from numpy arrays.

    Args:
        x (np.ndarray) : input vector with sample count as first dimension.
        y (np.ndarray) : label vector with sample count as first dimension.
        batch_size (int) : size of each minibatch.
    
    Returns:
        shuffled and batched tf.data.Dataset
    """
    sample_count = x.shape[0]
    #print(x)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #print(dataset)
    dataset = dataset.shuffle(sample_count+1)
    #print(dataset)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    #print(dataset)
    return dataset

def get_sample_count(*arrays):
    """Test variable number of arrays for matching sample count

    Args:
        *arrays (np.ndarrays) : arrays with sample count as first dimension.
    
    Returns:
        number of samples if each array in arrays has the same number of samples,
        else None.
    """
    print(len(arrays))
    for array in arrays:
        print(array.shape[0] == arrays[0].shape[0])
        print(array.shape)
    if all(array.shape[0] == arrays[0].shape[0] for array in arrays):
        sample_count = arrays[0].shape[0]
    else:
        sample_count = None
    return sample_count

def train_val_split(*arrays, split_val=.2):
    """Split arrays into training and validation sets.

    Args:
        *arrays (np.ndarray) : variable number of numpy arrays to be split.
        split_val (float) : 0 <= split_value <= 1. What percent of the arrays
            to put in the training set. (1-split_val is percent in validation set.)
    
    Returns:
        train_set, val_set (tuple(list(np.ndarray), list(np.ndarray))) : the input arrays
        split into train and validation sets.
    """
    sample_count = get_sample_count(*arrays)
    if not sample_count:
        raise Exception("Batch Axis inconsistent. All input arrays must have first axis of equal length.")
    arrays = random_shuffle(*arrays)
    split_idx = math.floor(sample_count * split_val)
    train_set = [array[split_idx:] for array in arrays]
    val_set = [array[:split_idx] for array in arrays]
    if len(train_set) == 1 and len(val_set) == 1:
        train_set = train_set[0]
        val_set = val_set[0]
    return train_set, val_set

def load_from_file(path, single_line=-1):
    try:
        data = pd.read_csv(path)#.values.astype(np.float32)
        #print(data)
        data_np = data.values.tolist()
        #print(data_np[0])
        if single_line!=-1:
            data_out = data_np[36*single_line:36*(single_line+1)]
            for i in range(100):
                data_out.append(data_out[i%len(data_out)])
        #print(data_out[0])
        #print(len(data_out))
        #attributes = ['k','QQ', 'x_b', 'phi_x', 'F', 'errF']
        #data_cut = 
        #for i in range(len(data_out)):
            #print(len(data_out[i]))
        return np.array(data_out)
    except:
        raise ValueError(f"Error loading file {path}")
    
def full_data_load(seed=231, line_num=-1):
    if seed:
        np.random.seed(seed)
    dataset = np.zeros((1,14), dtype=np.float32)
    data_folder = utils.package_relative_path('data/all')
    print('Linenum: ', line_num)
    for file in os.listdir(data_folder):
        if file[-3:] == 'csv':
            full_path = os.path.join(data_folder, file)
            data = load_from_file(full_path, single_line=line_num)
            dataset = np.concatenate([dataset, data], axis=0)
    print('-')
    print(len(dataset))
    dataset = dataset[1:,...]
    config_path = utils.package_relative_path('config.ini')
    config = utils.parse_config(config_path)['DATA'] #
    if config['factorization_cuts']:
        dataset = factorization_cuts(dataset)
    if config['linearization']:
        dataset = linearization(dataset)
    
    train_set, val_set = train_val_split(dataset)
    return train_set, val_set

def linearization(data):
    #id, xbj, t, Q2, k0, phi, L, sigma, error
    data[:,3] = np.log10(data[:,3]) # xbj = log10(xbj)
    data[:,2] = np.log10(data[:,2]) # Q2 = log10(Q2)
    return data

def factorization_cuts(data):
    #id, xbj, t, Q2, k0, phi, L, sigma, error
    data = data[np.where(data[:,2] > 1.5)] # Q2 > 1.5 GeV2
    data = data[np.where((-data[:,4] / data[:,2]) < .2 )] # -t/Q2 < .2
    return data

def compare_train_val():
    train, val = full_data_load()
    train_means = np.mean(train, axis=0)
    val_means = np.mean(val, axis=0)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn')
    args = parser.parse_args()
    eval(args.fn)()


