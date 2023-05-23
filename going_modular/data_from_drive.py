# Imports
import os
import zipfile
from pathlib import Path
import requests


# Mount the GDrive
from google.colab import drive
drive.mount('/content/drive')

data_path = Path("data/")  # The trailing forward slash (/) in the string
# is used to indicate that it represents a
# directory rather than a specific file.
# It's a common convention to include the trailing slash
# in directory paths to differentiate them from file paths.

image_path_full = data_path / "pizza_steak_sushi_full"

if image_path_full.is_dir():
    print(f"{image_path_full} directory exists.")
else:
    print(f"Did not find {image_path_full} directory, creating one...")
    image_path_full.mkdir(parents=True, exist_ok=True)
zip_data = "/content/drive/Othercomputers/My MacBook Air/GitHub/-Machine_Learning/Learning_Pytorch/pizza_steak_sushi_100_percent.zip"

with zipfile.ZipFile(zip_data, "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(image_path_full)

# Setup Dirs
train_dir_full = image_path_full / "train"
test_dir_full = image_path_full / "test"

# walk_through_dir(image_path)
