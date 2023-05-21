#Imports
import os 
import zipfile 
from pathlib import Path
import requests 


# Mount the GDrive
from google.colab import drive
drive.mount('/content/drive')

data_path = Path("data/") # The trailing forward slash (/) in the string 
                          # is used to indicate that it represents a 
                          # directory rather than a specific file. 
                          # It's a common convention to include the trailing slash 
                          # in directory paths to differentiate them from file paths.

image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
zip_data = "/content/drive/Othercomputers/My MacBook Air/GitHub/-Machine_Learning/Learning_Pytorch/pizza_steak_sushi_100_percent.zip"

with zipfile.ZipFile(zip_data, "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)

# Setup Dirs
train_dir = image_path / "train"
test_dir = image_path / "test"

walk_through_dir(image_path)