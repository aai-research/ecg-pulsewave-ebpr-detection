"""
Training utility module containing helper functions for optimization and setup.
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, SGD, RMSprop


def enable_reproducibility(random_seed):
    """
    Sets random seeds across various frameworks to ensure reproducible runs.
    Source: https://github.com/NVIDIA/framework-determinism

    Args:
        random_seed (int): The seed value to use for reproducibility.
    """

    if random_seed is not None:
        tf.config.experimental.enable_op_determinism()
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)


def get_optimizer(name, learning_rate, learning_rate_decay=None):
    """
    Instantiate and return a Keras optimizer based on the specified parameters.

    Args:
        name (str): The name of the optimizer (e.g., 'adam', 'sgd').
        learning_rate (float): The learning rate.
        learning_rate_decay (float, optional): The decay rate for learning rate (if supported). Defaults to None.

    Returns:
        tensorflow.keras.optimizers.Optimizer: The instantiated optimizer object.
    """
    if name == 'adam':
        if learning_rate_decay is not None:
            return Adam(learning_rate=learning_rate, decay=learning_rate_decay)
        else:
            return Adam(learning_rate=learning_rate)
    elif name == 'adadelta':
        return Adadelta(learning_rate=learning_rate)
    elif name == 'adagrad':
        return Adagrad(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    elif name == 'sgd':
        return SGD(learning_rate=learning_rate)
    else:
        raise Exception('Unknown optimizer. Aborting.')


def derive_class_weight(y_train):
    """
    Calculate class weights based on label frequencies to handle class imbalance.

    Args:
        y_train (np.ndarray): Array of training labels (one-hot encoded).

    Returns:
        dict: A dictionary mapping class indices to their computed weights.
    """
    cw = {i: 1 - (np.sum(y_train[..., i]) / len(y_train)) for i in range(len(y_train[0]))}
    return cw
