"""Custom callbacks for Keras model training."""
import gc
import logging
import os
from abc import abstractmethod
from tensorflow.keras.callbacks import Callback

from utils.metrics import calculate_metrics_for_classification
from utils.data import save_dict_as_json


class BasicCallback(Callback):
    """
    A basic callback that performs validation and saves results per epoch.
    """
    def __init__(self, x_val, y_val, record_ids_val, experiment_logdir, metrics=None, calculation_methods=None, **kwargs):
        """
        Initialize the BasicCallback.

        Args:
            x_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation target labels.
            record_ids_val (list): List of record IDs corresponding to validation data.
            experiment_logdir (str): Directory where logs and results will be saved.
            metrics (list, optional): List of metrics to evaluate.
            calculation_methods (list, optional): Methods for calculating metrics.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        os.makedirs(experiment_logdir, exist_ok=True)

        self.metrics = metrics
        self.calculation_methods = calculation_methods
        self.x_val = x_val
        self.y_val = y_val
        self.y_pred = None
        self.experiment_logdir = experiment_logdir
        self.record_ids_val = record_ids_val
        self.all_weights = {}
        self.all_results = {}

    def on_epoch_end(self, epoch, logs=None):
        """
        Actions to perform at the end of each epoch.

        Args:
            epoch (int): The current epoch index.
            logs (dict, optional): Dictionary of logs.
        """
        # Perform garbage collection
        gc.collect()

        # Predict on current state of model
        self.y_pred = self.model.predict(self.x_val)

        # Save model weights to memory instead of writing to disk every epoch
        self.all_weights[epoch] = self.model.get_weights()

        # Run specific code for model type
        self.on_epoch_end_specific(epoch=epoch, logs=logs)

    def save_all_results_to_json(self):
        """
        Save all accumulated results across epochs to a JSON file.
        """
        path = '{}/results_all.json'.format(self.experiment_logdir)
        save_dict_as_json(self.all_results, path)
        logging.debug('Saved all epoch results to {}.'.format(path))

    @abstractmethod
    def on_epoch_end_specific(self, epoch, logs=None):
        """
        Model-specific actions to perform at the end of each epoch. Must be implemented by subclasses.

        Args:
            epoch (int): The current epoch index.
            logs (dict, optional): Dictionary of logs.
        """
        pass


class ClassificationCallback(BasicCallback):
    """
    A callback tailored for classification tasks, calculating classification metrics at the end of each epoch.
    """
    def __init__(self, x_val, y_val, y_classes, record_ids_val, experiment_logdir, metrics=None, calculation_methods=None, **kwargs):
        """
        Initialize the ClassificationCallback.

        Args:
            x_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation target labels.
            y_classes (int): Number of target classes.
            record_ids_val (list): List of record IDs for validation data.
            experiment_logdir (str): Directory for saving logs and results.
            metrics (list, optional): Classification metrics to evaluate.
            calculation_methods (list, optional): Methods for evaluating metrics (e.g., sample vs subsample).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(x_val, y_val, record_ids_val, experiment_logdir, metrics=metrics, calculation_methods=calculation_methods, **kwargs)

        self.y_classes = y_classes

        if self.metrics is None:
            self.metrics = ['sensitivity', 'specificity']

        if self.calculation_methods is None:
            self.calculation_methods = ['sample_level', 'subsample_level']

    def on_epoch_end_specific(self, epoch, logs=None):
        """
        Calculate and record classification performance metrics at the end of the epoch.

        Args:
            epoch (int): The current epoch index.
            logs (dict, optional): Dictionary of logs.
        """
        # Calculate performance metrics
        result = calculate_metrics_for_classification(self.y_val, self.y_pred, self.y_classes, self.record_ids_val, self.metrics, self.calculation_methods)

        # Save epoch result to memory
        self.all_results['e{}'.format(epoch)] = result
