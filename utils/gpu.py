"""GPU configuration utilities."""
import os


def set_visible_gpu(gpu_id):
    """
    Set the specific GPU to be visible to TensorFlow/Keras.

    Args:
        gpu_id (int or str): The ID of the GPU to make visible.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
