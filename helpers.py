# Recreating this file for practice
"""
A series of helper functions used throughout the course.
    
If a function gets defined once and could be used over and over, it'll go in here.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path
import requests

# Walk through an image classification directory and find out hoe many files (images)
# are in each subdirectory.


def walk_thorugh_dir(dir_path):
    """
    Walk through dir_path returning its contents
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
        number of subdirectories in dir_path
        number of images (files) in each subdirectory
        number of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots decison boundaries of model predicting on X in comprison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup predictoin boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, Y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjusts logits to prediction labels
    if 



















# This is a test in dual monitor setup
