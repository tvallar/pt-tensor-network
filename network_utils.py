import configparser
import os
import shutil
from termcolor import colored
import io

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import xsx

def parse_config(config_path):
    """Parse config.ini and convert to proper datatypes

    Whenever a parameter is added to config.ini, be sure to add a type
    conversion. See src code for examples.

    Returns:
        A dictionary with the same structure as the config file, but
        correct types.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    #print(config.sections())
    converted_config = {s:dict(config.items(s)) for s in config.sections()}
    #print(converted_config)

    def change_key_to_type(section, key, type_to_cast):
        nonlocal converted_config
        #print(section, ' ', key)
        entry_to_change = converted_config[section][key]
        if type_to_cast == bool:
            if entry_to_change == 'True':
                converted_config[section][key] = True
            elif entry_to_change == 'False':
                converted_config[section][key] = False
            else:
                raise ValueError(f"Config file error. [{section}][{key}] should be 'True' or 'False'. Is currently {entry_to_change}.")
        else:
            converted_config[section][key] = type_to_cast(entry_to_change)

    change_key_to_type('TRAINING', 'learning_rate', float)
    change_key_to_type('TRAINING', 'clip_norm', float)
    change_key_to_type('TRAINING', 'batch_size', int)
    change_key_to_type('TRAINING', 'optimizer', str)
    change_key_to_type('TRAINING', 'save_freq', int)

    change_key_to_type('NETWORKS', 'weight_norm_penalty', float)
    change_key_to_type('NETWORKS', 'dropout_fraction', float)
    converted_config['NETWORKS']['network_hidden'] = eval(converted_config['NETWORKS']['network_hidden'])
    #converted_config['TRAINING']['xsx_function'] = eval(f"xsx.{converted_config['TRAINING']['xsx_function']}")


    change_key_to_type('PREDICTION', 'sample_models', int)

    change_key_to_type('DATA', 'factorization_cuts', bool)
    change_key_to_type('DATA', 'linearization', bool)

    change_key_to_type('AE', 'batch_size', int)
    change_key_to_type('AE', 'epochs', int)
    change_key_to_type('AE', 'save_frequency', int)

    return converted_config

def form_factor_figure(ffs):
    """Visualize form factor predicitons.

    Args:
        ffs (tf.Tensor) : (batch_size, 8) output of model.
    
    Returns:
        pyplot imshow of ffs tensor.
    """
    figure = plt.figure()
    plt.imshow(ffs, cmap=mpl.cm.get_cmap(name='Blues'))
    plt.title('Form Factor Predictions')
    plt.colorbar()
    labels = ['reH', 'imH', 'reE', 'imE', 'reHt', 'imHt', 'reHt', 'imEt']
    plt.xticks(np.arange(len(labels)),labels, rotation=45)
    print(type(ffs.shape[0]))
    plt.yticks(np.arange(ffs.get_shape().as_list()[0]))
    plt.xlabel('Form Factors')
    plt.ylabel('Sample')
    return figure

def make_results_dirs(run_name, base_path='saves'):
    """ Create directory to save logs & checkpoints

    Creates new directory for a training run. Adds unique number
    to end of run_name if run_name has already been used.

    Args:
        run_name (str): Name of this experiment/training run.
        base_path (str, optional): name of root directory to put
             this run's folder in. Will be created if it doesn't already exist.
    
    Returns:
        log directory (str)
        checkpoint directory (str)
   """
    base_dir = os.path.join('saves/', run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir)
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir)
    return log_dir, checkpoint_dir

def tensor_to_colored_string(tensor, color):
    """Print tensor's numerical value in colored ascii.

    Args:
        tensor (tf.Tensor or tf.keras.metrics.Metric) : Tensor to be printed.
        color (str) : options are grey, red, green, yellow, blue, magenta, cyan
            and white.
    
    Returns:
        None
    """
    if isinstance(tensor, tf.keras.metrics.Metric):
        val = tensor.result().numpy()
    else:
        val = tensor.numpy()
    return colored(val, color)

def tensor_to_image(tensor):
    """Convert Tensorflow matrix to standard image format (for plotting)

    Pads rank 2 Tensor with sample and channel dimensions.

    Args:
        tensor (tf.Tensor) : rank 2 Tensor.
    
    Returns:
        rank 4 tensor
    """ 
    return tf.expand_dims(tf.expand_dims(tensor, 0), -1)

def figure_to_image(figure):
    """Convert pyplot figure to image.

    Args:
        figure (pyplot figure)
    
    Returns:
        image of figure in a tf.Tensor.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def package_relative_path(path):
    return os.path.join(
        os.path.dirname(__file__),
        path
    )
