try: 
    import os
    from PIL import Image
    from torchvision.transforms import Compose, ToTensor, Resize
    import torch
    import torch.nn as nn
    import torchvision
    import torch.nn.functional as F
    from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor
    from torch.utils.data import RandomSampler, DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    import random
    import pickle
    from Training import *
    import sys

except Exception as e:
    print(f"Error: {e}")
    print("Please use python 3.11 and install the required libraries using requirements.txt")
    exit(1)

python_version = sys.version.split()[0]
python_major_version = python_version.split('.')[0]
python_minor_version = python_version.split('.')[1]

if python_major_version != '3' or python_minor_version != '11':
    print(f'Your python version is: {python_version}')
    print("Please use Python 3.11")
    exit(1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

DATA_DIR = "Fundus_TestDataSet"
BATCH_SIZE = 4 
PRINT_ALL = True
PRINT_EVERY_N = 10
RES_Y = 400
RES_X = 640

def print_syntax():
    print("Usage: python Test.py [model_file] [data_dir]")
    print("   model_file: The model file to use for testing")
    print("   data_dir: The directory containing the test data")
    print("   Example: python Test.py model_file Fundus_TestDataSet")

# If model specified on commandline, use it
# Otherwise, use the default model

model_file = 'model_file'
if len(sys.argv) > 1:
    model_file = sys.argv[1]

# If file path doesn't include models/ prefix and file doesn't exist without it, add it
if not os.path.exists(model_file) and os.path.exists('models/' + model_file):
    model_file = 'models/' + model_file

if not os.path.exists(model_file):
    print(f"Model file {model_file} does not exist")
    print_syntax()
    exit(1)

# If model name ends with _torch, use torch model
if model_file.endswith('_torch'):
    loaded_model = torch.load(model_file, map_location=torch.device(device))

    '''
    # Convert to pickle
    loaded_model_pickle = pickle.dumps(loaded_model)
    # Save the pickle model
    with open(model_file + '_pickle', 'wb') as f:
        f.write(loaded_model_pickle)
    '''
else:
    with open(model_file, 'rb') as f:
        model_pickle = f.read()
    loaded_model = pickle.loads(model_pickle)

# If data directory specified on commandline, use it
if len(sys.argv) > 2:
    DATA_DIR = sys.argv[2]  

if not os.path.exists(DATA_DIR):
    print(f"Data directory {DATA_DIR} does not exist")
    print_syntax()
    exit(1)

# Preprocess the test data
main_preprocess(DATA_DIR, DATA_DIR, device)

# Load the test data
test_dataset = FundusDataset(DATA_DIR, device, include_orig=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loaded_model.eval()
total_images = 0
with torch.no_grad():
        # Graph the results
        for i, (images, masks, original_images) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = loaded_model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
        
            num_images = images.size(0)

            for idx in range(num_images):
                total_images += 1
                if not PRINT_ALL and total_images % PRINT_EVERY_N != 0:
                    continue
                
                fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
                # Original Image
                axes[0].imshow(original_images[idx].cpu().permute(1, 2, 0))
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # Sobel Filter Output
                axes[1].imshow(images[idx].squeeze().cpu(), cmap='gray')
                axes[1].set_title("Sobel Filter Output")
                axes[1].axis('off')
                
                # Ground Truth Mask
                axes[2].imshow(masks[idx].cpu().squeeze(), cmap='gray')
                axes[2].set_title("Ground Truth Mask")
                axes[2].axis('off')
                
                # Predicted Mask in Greyscale
                axes[3].imshow(outputs[idx].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[3].set_title("Predicted Mask (Greyscale)")
                axes[3].axis('off')

                # Predicted Mask in Binary
                outputs_b = (outputs > 0.5).float()
                axes[4].imshow(outputs_b[idx].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[4].set_title("Predicted Mask (Binary)")
                axes[4].axis('off')
                
                plt.tight_layout()
                plt.show()

