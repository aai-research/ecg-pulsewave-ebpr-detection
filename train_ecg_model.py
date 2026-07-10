"""Train a 1D-CNN for eBPR detection from 12-lead resting ECG signals."""

from utils.gpu import set_visible_gpu

set_visible_gpu(0)

import config
from utils.training import enable_reproducibility, get_optimizer, derive_class_weight

enable_reproducibility(config.seed)

import gc
import logging
import os
import shutil
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout, Softmax, Activation

from utils.callbacks import ClassificationCallback
from utils.loader import load_timeseries_data
from utils.evaluation import evaluate_experiment
from utils.loader import load_ecg_labels, find_records
from utils.logger import initialize_logger
from utils.splitting import ecg_patients_split_cross_validation


def load_data(data_folder, window_size, subsampling_factor, fold, num_leads=12, sampling_rate=500, num_cls=2):
    """Load ECG data for training and validation."""
    labels = load_ecg_labels()
    record_ids = find_records(data_folder)

    S = []

    for i, recid in enumerate(record_ids):
        c = labels[f"{recid}.npy"]
        S.append([c])

    logging.info('Calculating stratified splits...')
    recids_train, recids_val, S_train, S_val = ecg_patients_split_cross_validation(recids=np.array(record_ids),
                                                                                   labels=np.array(S),
                                                                                   fold=fold)

    logging.info('Loading ECGs...')
    X_train, Y_train, R_train = load_timeseries_data(recids_train, labels, subsampling_factor, window_size, data_folder,
                                                     num_leads, sampling_rate, num_cls, is_ecg=True)
    X_val, Y_val, R_val = load_timeseries_data(recids_val, labels, subsampling_factor, window_size, data_folder,
                                               num_leads, sampling_rate, num_cls, is_ecg=True)

    return X_train, X_val, Y_train, Y_val, R_train, R_val, (window_size, num_leads), num_cls


def train_model(data_folder, params, fold, verbose=True):
    """Train the ECG model for a specific fold."""
    initialize_logger({True: 'INFO', False: 'ERROR'}[verbose])

    logdir = params['logdir']

    logging.info('Loading data...')
    X_train, X_val, Y_train, Y_val, R_train, R_val, input_shape, label_shape = load_data(data_folder=data_folder,
                                                                                         window_size=params['window_size'],
                                                                                         subsampling_factor=params[
                                                                                             'subsampling_factor'],
                                                                                         fold=fold,
                                                                                         num_leads=params['num_leads'],
                                                                                         num_cls=len(params['y_classes'])
                                                                                         )

    logging.info('Cleaning logdir...')
    shutil.rmtree(logdir, ignore_errors=True)

    logging.info('Building model...')
    model = Sequential()

    for i in range(len(params['kernel_sizes_conv'])):
        current_kernel_size = params['kernel_sizes_conv'][i]
        if i == 0:
            model.add(Conv1D(kernel_size=current_kernel_size,
                             kernel_initializer=params['kernel_initializer_conv'],
                             filters=params['filters_conv'],
                             padding=params['padding_conv'],
                             input_shape=input_shape))
        else:
            model.add(Conv1D(kernel_size=current_kernel_size,
                             kernel_initializer=params['kernel_initializer_conv'],
                             filters=params['filters_conv'],
                             padding=params['padding_conv']))

        # Activation function
        model.add(Activation(params['activation_conv']))

        # Max-pooling layer
        model.add(MaxPool1D(strides=params['strides_pool_conv'], pool_size=params['size_pool_conv']))

        # Dropout (except after input layer)
        if i != 0 and params['dropout_conv'] > 0:
            model.add(Dropout(params['dropout_conv']))

    # Transition to dense layers
    model.add(GlobalAveragePooling1D())

    # Dense layer groups
    for i in range(params['groups_dense']):
        # Dense layer
        model.add(Dense(units=params['units_dense'], kernel_initializer=params['kernel_initializer_dense']))

        # Activation function
        model.add(Activation(params['activation_dense']))

        # Dropout
        model.add(Dropout(params['dropout_dense']))

    # Output layer (units = width of label)
    model.add(Dense(units=label_shape, kernel_initializer=params['kernel_initializer_dense']))

    # Softmax activation
    model.add(Softmax())

    logging.info('Starting training...')
    custom_callback = ClassificationCallback(x_val=X_val,
                                             y_val=Y_val,
                                             y_classes=params['y_classes'],
                                             record_ids_val=R_val,
                                             metrics=params['metrics'],
                                             calculation_methods=params['calculation_methods'],
                                             experiment_logdir=logdir)

    optimizer = get_optimizer(name=params['optimizer'],
                              learning_rate=params['learning_rate'],
                              learning_rate_decay=params['learning_rate_decay'])

    model.compile(optimizer=optimizer, loss=params['loss_function'])

    model.fit(x=X_train,
              y=Y_train,
              batch_size=params['batch_size'],
              epochs=params['number_epochs'],
              shuffle=params['shuffle'],
              validation_data=(X_val, Y_val),
              callbacks=[custom_callback],
              class_weight=derive_class_weight(Y_train)
              )

    custom_callback.save_all_results_to_json()

    logging.info('Training complete.')
    best_e = evaluate_experiment(logdir=logdir,
                                 y_labels=params['y_classes'],
                                 calculation_methods=params['calculation_methods'],
                                 metrics=params['metrics'],
                                 target_metric=params['target_metric'],
                                 metric_thresholds=params['metric_thresholds'],
                                 save_raw_results=True,
                                 epochs=params['number_epochs'])

    # Save the model.
    model_folder = os.path.dirname(logdir)
    save_models(custom_callback, best_e, model_folder, params['target_calc_method'])
    logging.info('Saved model.')

    with open(f'{model_folder}/params.json', 'w') as f:
        json.dump(params, f, indent=4)

    # Perform garbage collection
    clear_session()
    gc.collect()


def save_models(custom_callback, best_e, model_folder, target_calc_method):
    """Save the best model architecture and weights."""
    os.makedirs(model_folder, exist_ok=True)
    
    with open('{}/model.json'.format(model_folder), 'w') as f:
        f.write(custom_callback.model.to_json())
        
    best_epoch = best_e[target_calc_method]
    custom_callback.model.set_weights(custom_callback.all_weights[best_epoch])
    custom_callback.model.save_weights('{}/weights.h5'.format(model_folder))


if __name__ == '__main__':
    parent_dir = f"{config.directory}/models"
    model_type = 'ecg'

    kernel_sizes_conv = [15, 15, 11, 11, 7, 7, 5]
    filters_conv = 64
    dropout_conv = 0.1
    groups_dense = 3
    units_dense = 128
    dropout_dense = 0.5
    lr = 0.001
    bs = 64

    kernel_str = "-".join([str(x) for x in kernel_sizes_conv])
    model_name = f"{kernel_str}_{filters_conv}_{dropout_conv}_{groups_dense}_{units_dense}_{dropout_dense}_{lr}_{bs}"

    for fold in range(1, config.num_folds + 1):

        parameters = {'kernel_sizes_conv': kernel_sizes_conv,
                  'kernel_initializer_conv': 'he_uniform',
                  'filters_conv': filters_conv,
                  'padding_conv': 'same',
                  'activation_conv': 'elu',
                  'strides_pool_conv': 2,
                  'size_pool_conv': 2,
                  'dropout_conv': dropout_conv,
                  'groups_dense': groups_dense,
                  'units_dense': units_dense,
                  'activation_dense': 'elu',
                  'dropout_dense': dropout_dense,
                  'kernel_initializer_dense': 'he_uniform',
                  'number_epochs': 200,
                  'optimizer': 'adam',
                  'learning_rate': lr,
                  'learning_rate_decay': None,
                  'batch_size': bs,
                  'loss_function': 'categorical_crossentropy',
                  'shuffle': True,
                  'subsampling_factor': 5,
                  'window_size': 2000,
                  'num_leads': 12,
                  'seed': config.seed,
                  'y_classes': ['FALSE', 'TRUE'],
                  'calculation_methods': ['sample_level', 'subsample_level'],
                  'metrics': ['sensitivity', 'specificity', 'AUC', 'MCC'],
                  'metric_thresholds': {'sensitivity': 0.9, 'specificity': 0.9},
                  'target_metric': 'MCC',
                  'target_calc_method': "sample_level",
                  'logdir': f'{parent_dir}/{model_type}/{model_name}/{fold}/logs'
                  }
        train_model(data_folder=f'./{config.directory}/data/train/{model_type}', params=parameters, fold=fold)
