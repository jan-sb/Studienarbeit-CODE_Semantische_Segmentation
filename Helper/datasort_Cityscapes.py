from torchvision.datasets import Cityscapes
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
import os


dataset = Cityscapes(root='Cityscapes/', split='val', mode='fine', target_type='semantic')
print(len(dataset))


# Create a figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

# Plot the first image
axes[0].imshow(dataset[0][0])
axes[0].axis('off')  # Turn off axis for cleaner display

# Plot the second image
axes[1].imshow(dataset[0][1], cmap='gray')
axes[1].axis('off')  # Turn off axis for cleaner display

# Save the figure
plt.savefig('Daten/CityscapesDaten/Citytest.png', dpi=300)  # Adjust dpi as needed


print(type(dataset[0][0]))

Path("Daten/CityscapesDaten/images").mkdir(parents=True, exist_ok=True)
Path("Daten/CityscapesDaten/semantic").mkdir(parents=True, exist_ok=True)

total = len(dataset)

for i in range(len(dataset)):
    print(f'Processing image {i}/{total}')
    # Get the image and its semantic map
    image, semantic_map = dataset[i]

    # Save the image and its semantic map
    image.save(f"Daten/CityscapesDaten/images/00000{i+2975}_01.png")
    semantic_map.save(f"Daten/CityscapesDaten/semantic/00000{i+2975}_01.png")