from __future__ import print_function, absolute_import, division
import cv2 as cv
import numpy as np
import sys
import datetime
import os
from cityscapesscripts.helpers.labels import labels
from PIL import Image
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
import json
from collections import namedtuple
from cityscapesscripts.helpers.labels import *



def watershed_segmentation(image):
    assert image is not None, "file could not be read, error in watershed segmentation"
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    segmented = np.zeros_like(image)
    for i in range(2, markers.max() + 1):
        segmented[markers == i] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

    segmented = cv.addWeighted(image, 1, segmented, 0.3, gamma=0)

    image[markers == -1] = [255, 0, 0]
    markers = np.uint8(markers)

    return segmented, markers



def cap_def(path):
    cap = cv.VideoCapture(path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    return cap, width, height, length, fps




def update_console(message):
    os.system('cls' if os.name == 'nt' else 'clear')
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()



def video_writer(output_path, fps, resolution_tuple):
    video_title = f'video_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_path_final = os.path.join(output_path, video_title)
    out = cv.VideoWriter(output_path_final, fourcc, fps, resolution_tuple)
    print(f'Video writer initialized: {output_path}/{video_title}, with {fps} fps and resolution {resolution_tuple}, fourcc: {fourcc}')
    return out

def update_progress_bar(current_frame, max_frames, bar_length=20):
    ratio = current_frame / max_frames
    progress = int(bar_length * ratio)
    bar = "[" + "=" * progress + " " * (bar_length - progress) + "]"
    percentage = int(ratio * 100)
    return f"{bar} {percentage}%"


def analyse_dataset_RGB(path):
    classes = {
        (128, 64, 128): 'road',
        (244, 35, 232): 'sidewalk',
        (70, 70, 70): 'building',
        (102, 102, 156): 'wall',
        (190, 153, 153): 'fence',
        (153, 153, 153): 'pole',
        (250, 170, 30): 'traffic light',
        (220, 220, 0): 'traffic sign',
        (107, 142, 35): 'vegetation',
        (152, 251, 152): 'terrain',
        (70, 130, 180): 'sky',
        (220, 20, 60): 'person',
        (255, 0, 0): 'rider',
        (0, 0, 142): 'car',
        (0, 0, 70): 'truck',
        (0, 60, 100): 'bus',
        (0, 80, 100): 'train',
        (0, 0, 230): 'motorcycle',
        (119, 11, 32): 'bicycle',
        (0, 0, 0): 'unlabeled'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the counts
    counts = {class_name: [] for class_name in classes.values()}      
    image_names = []

    def count_classes(image_path):
        # Initialize the counts for this image
        image_counts = {class_name: 0 for class_name in classes.values()}
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert('RGB')
        # Calculate the total number of pixels for this image
        total_pixels = np.prod(image.size)
        # Convert the image to a PyTorch tensor and send it to the device
        image_tensor = torch.from_numpy(np.array(image)).to(device)
        # Iterate over each pixel in the image
        for rgb in classes.keys():
            # Create a mask for the current class
            mask = (image_tensor == torch.tensor(rgb, device=device)).all(dim=2)
            # Count the pixels in the mask and add them to the count for the current class
            image_counts[classes[rgb]] += mask.sum().item() / total_pixels
        # Add the counts for this image to the overall counts
        for class_name in classes.values():
            counts[class_name].append(image_counts[class_name])
        # Store the image name
        image_names.append(os.path.basename(image_path))

    # Get the total number of images in the directory
    total_images = len(os.listdir(path))
    
    # Iterate over all the image files in the directory
    for i, image_file in enumerate(os.listdir(path)):
        # Print the progress
        print(f'Analyzing image {i}/{total_images}')
        # Count the classes in the image
        count_classes(os.path.join(path, image_file))

    # Create the DataFrame
    data = []
    for i, image_name in enumerate(image_names):
        for class_name in classes.values():
            data.append({"Image": image_name, "Class": class_name, "Pixel Count": counts[class_name][i]})
    df = pd.DataFrame(data)

    return df


def analyse_dataset_GRAY(path):
    classes = {
        7: 'road',
        8: 'sidewalk',
        11: 'building',
        12: 'wall',
        13: 'fence',
        17: 'pole',
        19: 'traffic light',
        20: 'traffic sign',
        21: 'vegetation',
        22: 'terrain',
        23: 'sky',
        24: 'person',
        25: 'rider',
        26: 'car',
        27: 'truck',
        28: 'bus',
        31: 'train',
        32: 'motorcycle',
        33: 'bicycle',
        0: 'unlabeled'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the counts
    counts = {class_name: [] for class_name in classes.values()}      
    image_names = []

    def count_classes(image_path):
        # Initialize the counts for this image
        image_counts = {class_name: 0 for class_name in classes.values()}
        # Open the image and convert it to grayscale
        image = Image.open(image_path).convert('L')
        # Calculate the total number of pixels for this image
        total_pixels = np.prod(image.size)
    # Convert the image to a PyTorch tensor and send it to the device
        image_tensor = torch.from_numpy(np.array(image)).to(device)
        # Iterate over each pixel in the image
        for rgb in classes.keys():
            # Create a mask for the current class
            mask = (image_tensor == torch.tensor(rgb, device=device))
            # Count the pixels in the mask and add them to the count for the current class
            image_counts[classes[rgb]] += mask.sum().item() / total_pixels
        # Add the counts for this image to the overall counts
        for class_name in classes.values():
            counts[class_name].append(image_counts[class_name])
        # Store the image name
        image_names.append(os.path.basename(image_path))

    # Get the total number of images in the directory
    total_images = len(os.listdir(path))
    
    # Iterate over all the image files in the directory
    for i, image_file in enumerate(os.listdir(path)):
        # Print the progress
        print(f'Analyzing image {i}/{total_images}')
        # Count the classes in the image
        count_classes(os.path.join(path, image_file))

    # Create the DataFrame
    data = []
    for i, image_name in enumerate(image_names):
        for class_name in classes.values():
            data.append({"Image": image_name, "Class": class_name, "Pixel Count": counts[class_name][i]})
    df = pd.DataFrame(data)

    return df

def class_distribution_violin_plot(df, output):
    # Min-Max normalize the "Pixel Count" values for each class
    df['Pixel Count'] = df.groupby('Class')['Pixel Count'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Create a horizontal violin plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="Pixel Count", y="Class", data=df, orient='h', cut=0, inner='quart', density_norm='count')
    plt.title('Distribution of min-max normalized pixel counts per image for each class')
    plt.tight_layout()
    plt.savefig(f'Daten/{output}/pixel_count_distribution.png')

def stratified_kfold_and_violin_plot(df, output, k=5):
    # Get the class for each image
    image_classes = df.groupby('Image')['Class'].first().values

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Add a new column to the original DataFrame to indicate the source
    df['Source'] = 'Original'

    # Split the images into k groups
    for i, (train_index, test_index) in enumerate(skf.split(df['Image'].unique(), image_classes)):
        # Get the images for this group
        group_images = df['Image'].unique()[test_index]
        # Save the image names for this group to a CSV file
        pd.DataFrame(group_images, columns=['Image']).to_csv(f'Daten/{output}/group_{i}_images.csv', index=False)
        # Get the rows for this group
        group_rows = df[df['Image'].isin(group_images)].copy()  # Make a copy to avoid a SettingWithCopyWarning
        # Add a new column to the k-fold DataFrame to indicate the source
        group_rows['Source'] = f'Fold {i}'

        # Concatenate the original DataFrame and the k-fold DataFrame
        combined_df = pd.concat([df, group_rows])

        # Create a violin plot for the combined data
        plt.figure(figsize=(10, 5))
        sns.violinplot(x="Pixel Count", y="Class", hue="Source", split=True, data=combined_df, orient='h', cut=0, inner='quart', density_norm='count')
        plt.title(f'Comparison of original distribution and distribution for fold {i}')
        plt.tight_layout()
        plt.savefig(f'Daten/{output}/distribution_comparison_{i}.png')
        
        
def compare_distributions(df1, df2, output):
    # Add a new column to the DataFrames to indicate the source
    df1['Source'] = 'DataFrame 1'
    df2['Source'] = 'DataFrame 2'

    # Concatenate the DataFrames
    combined_df = pd.concat([df1, df2])

    # Create a violin plot for the combined data
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="Pixel Count", y="Class", hue="Source", split=True, data=combined_df, orient='h', cut=0, inner='quart', density_norm='count')
    plt.title('Comparison of distributions')
    plt.tight_layout()
    plt.savefig(f'Daten/{output}/distribution_comparison.png')
    
    
def save_tensor_as_png(tensor, filename):
    # Convert the tensor to a PIL Image
    image = transforms.ToPILImage()(tensor)

    # Save the PIL Image as a PNG
    image.save(filename)
    
def calculate_normalization_values(path):
    # Initialize lists to store all pixel values for each color channel
    r_values = []
    g_values = []
    b_values = []

    # Get a list of all files in the directory
    filenames = os.listdir(path)

    # Iterate over all files in the directory
    for filename in tqdm(filenames, desc="Processing images"):
        # Open the image file
        with Image.open(os.path.join(path, filename)) as img:
            # Convert the image to a tensor
            tensor = transforms.ToTensor()(img)
            # Split the tensor into color channels
            r, g, b = tensor
            # Add the pixel values to the lists
            r_values.append(r.flatten())
            g_values.append(g.flatten())
            b_values.append(b.flatten())

    # Concatenate all pixel values for each color channel
    r_values = torch.cat(r_values)
    g_values = torch.cat(g_values)
    b_values = torch.cat(b_values)

    # Calculate the mean and standard deviation for each color channel
    r_mean, r_std = round(r_values.mean().item(), 4), round(r_values.std().item(), 4)
    g_mean, g_std = round(g_values.mean().item(), 4), round(g_values.std().item(), 4)
    b_mean, b_std = round(b_values.mean().item(), 4), round(b_values.std().item(), 4)

    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)

def calculate_multi_normalization_values(paths, batch_size=50):
    # Initialize running totals and counts for each color channel
    r_total = g_total = b_total = 0
    count = 0

    # Iterate over all paths
    for path in paths:
        # Get a list of all files in the directory
        filenames = os.listdir(path)

        # Create a DataLoader to handle batching
        data_loader = DataLoader(filenames, batch_size=batch_size)

        # Iterate over all batches of files in the directory
        for batch in tqdm(data_loader, desc="Processing images"):
            # Iterate over all files in the batch
            for filename in batch:
                # Open the image file
                with Image.open(os.path.join(path, filename)) as img:
                    # Convert the image to a tensor
                    tensor = transforms.ToTensor()(img)
                    # Split the tensor into color channels
                    r, g, b = tensor
                    # Update running totals and count
                    r_total += r.sum().item()
                    g_total += g.sum().item()
                    b_total += b.sum().item()
                    count += tensor.numel() / 3  # Divide by 3 because there are 3 color channels

    # Calculate the mean for each color channel
    r_mean = round(r_total / count, 4)
    g_mean = round(g_total / count, 4)
    b_mean = round(b_total / count, 4)

    # Initialize running totals for the standard deviations for each color channel
    r_total = g_total = b_total = 0

    # Iterate over all paths again to calculate the standard deviations
    for path in paths:
        # Get a list of all files in the directory
        filenames = os.listdir(path)

        # Create a DataLoader to handle batching
        data_loader = DataLoader(filenames, batch_size=batch_size)

        # Iterate over all batches of files in the directory
        for batch in tqdm(data_loader, desc="Processing images"):
            # Iterate over all files in the batch
            for filename in batch:
                # Open the image file
                with Image.open(os.path.join(path, filename)) as img:
                    # Convert the image to a tensor
                    tensor = transforms.ToTensor()(img)
                    # Split the tensor into color channels
                    r, g, b = tensor
                    # Update running totals for the standard deviations
                    r_total += ((r - r_mean) ** 2).sum().item()
                    g_total += ((g - g_mean) ** 2).sum().item()
                    b_total += ((b - b_mean) ** 2).sum().item()

    # Calculate the standard deviation for each color channel
    r_std = round((r_total / count) ** 0.5, 4)
    g_std = round((g_total / count) ** 0.5, 4)
    b_std = round((b_total / count) ** 0.5, 4)
    
    data = {
        "mean": (r_mean, g_mean, b_mean),
        "std": (r_std, g_std, b_std)
    }

    # Save the data to a JSON file
    with open('Daten/mean_std.json', 'w') as f:
        json.dump(data, f)

    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)


import os

def create_model_directory(model, i):
    dir_name = f'Own_Weights/{model}_k_fold_{i}'
    os.makedirs(dir_name, exist_ok=True)
    
 
def create_ground_truth_json():
    label_dict = {}
    for label in labels:
        if label.trainId != -1:
            if label.trainId not in label_dict:
                label_dict[label.trainId] = []
                label_dict[label.trainId].append([label.id, label.name, str(label.color)])
            else:
                label_dict[label.trainId].append([label.id, label.name, str(label.color)]) 
    
    with open('Daten/label_dict.json', 'w') as f:
        json.dump(label_dict, f, indent=4, sort_keys=True)
    


def create_ground_truth(in_path, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a mapping from label ID to trainId
    id_to_trainId = {label.id: label.trainId for label in labels if label.trainId >= 0}
    id_to_trainId = torch.tensor([id_to_trainId.get(i, -1) for i in range(256)], dtype=torch.long, device=device)

    # Get the list of image files
    image_files = os.listdir(in_path)
    total = len(image_files)

    # Process the images
    for i, image_file in enumerate(image_files):
        # Load the image
        image = Image.open(os.path.join(in_path, image_file))
        # Convert the image to a PyTorch tensor and move it to the device
        image = torch.from_numpy(np.array(image)).to(device)
        # Convert the label IDs to trainId
        image = id_to_trainId[image.long()]
        # Convert the tensor back to a PIL image
        image = transforms.ToPILImage()(image.cpu().byte())
        # Save the image with the original name
        image.save(os.path.join(out_path, image_file))
        print(f'Processed image {i}/{total}')
        # if i > 10:
        #     break
    
    