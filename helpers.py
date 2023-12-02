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
