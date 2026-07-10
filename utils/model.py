"""
Model utility module for loading and modifying Keras models.
"""
import json
from tensorflow.keras.models import model_from_json


def remove_softmax_from_model(loaded_model_json):
    """
    Remove the Softmax layer from the model configuration and replace softmax activations with relu.

    Args:
        loaded_model_json (str): JSON string representation of the model.

    Returns:
        str: Modified JSON string representation of the model without Softmax.
    """
    # Convert JSONized model config to dictionary
    dct = json.loads(loaded_model_json)

    # Extract layer list
    layers = dct['config']['layers']

    # Remove Softmax layer if present
    for i, layer in enumerate(layers):
        if layer['class_name'] == 'Softmax':
            layers.pop(i)
            break

    # Convert dict back to json string
    loaded_model_json = json.dumps(dct)

    # Replace Softmax by ReLU activations if present
    loaded_model_json = loaded_model_json.replace('"activation": "softmax"', '"activation": "relu"')

    return loaded_model_json


def load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=False):
    """
    Load a Keras model architecture and its weights from file paths.

    Args:
        modelpath (str): Path to the model JSON file.
        weightspath (str): Path to the weights HDF5 file.
        remove_softmax (bool, optional): Whether to strip the Softmax layer. Defaults to False.

    Returns:
        tensorflow.keras.models.Model: The loaded Keras model.
    """
    with open(modelpath, 'r') as json_file:
        loaded_model_json = json_file.read()

    if remove_softmax:
        loaded_model_json = remove_softmax_from_model(loaded_model_json)

    model = model_from_json(loaded_model_json)
    model.load_weights(weightspath)

    return model


def load_models_from_paths(modelpath, weightspath):
    """
    Load a Keras model from paths and also load a variant with Softmax removed.

    Args:
        modelpath (str): Path to the model JSON file.
        weightspath (str): Path to the weights HDF5 file.

    Returns:
        tuple: A tuple containing the original model and the model with Softmax removed.
    """
    model = load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=False)
    model_wo_softmax = load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=True)

    return model, model_wo_softmax
