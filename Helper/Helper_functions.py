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
    # Define the classes
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

    # Get the total number of images
    total_images = len(os.listdir(path))

    # Iterate over all the image files in the directory
    for i, image_file in enumerate(os.listdir(path)):
        # Print the progress
        print(f'Analyzing image {i}/{total_images}')
        # Construct the full path of the image file
        image_path = os.path.join(path, image_file)
        # Count the classes in the image
        count_classes(image_path)
    
    # Prepare data for seaborn
    data = []
    for class_name, values in counts.items():
        for value in values:
            data.append({"Class": class_name, "Pixel Count": value})

    df = pd.DataFrame(data)
    
    data = []
    for i, image_name in enumerate(image_names):
        for class_name in classes.values():
            data.append({"Image": image_name, "Class": class_name, "Pixel Count": counts[class_name][i]})
    df2 = pd.DataFrame(data)

    return df, df2
    

def class_distribution_violin_plot(df):
     # Min-Max normalize the "Pixel Count" values for each class
    df['Pixel Count'] = df.groupby('Class')['Pixel Count'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Create a horizontal violin plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="Pixel Count", y="Class", data=df, orient='h', cut=0, inner='quart', density_norm='count')
    plt.title('Distribution of min-max normalized pixel counts per image for each class')
    plt.tight_layout()
    plt.savefig('Daten/pixel_count_distribution.png')
    
     # Specify the classes to include
    classes_to_include = ['road', 'person', 'rider', 'car', 'motorcycle', 'bike', 'unlabeled']

     # Normalize the "Pixel Count" values for each class
    df_filtered = df[df['Class'].isin(classes_to_include)]
        # Min-Max normalize the "Pixel Count" values for each class
    df_filtered['Pixel Count'] = df_filtered.groupby('Class')['Pixel Count'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Create a horizontal violin plot for the specified classes with normalized data
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="Pixel Count", y="Class", data=df_filtered, orient='h', cut=0, inner='quart', density_norm='width')
    plt.title('Distribution of min-max normalized pixel counts per image for selected classes')
    plt.tight_layout()
    plt.savefig('Daten/selected_classes_min_max_normalized_pixel_count_distribution.png')
    
def stratified_kfold_and_violin_plot(df, k=5):
    # Get the class for each image
    image_classes = df.groupby('Class')['Pixel'].first().values

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=k)

    # Split the images into k groups
    for i, (train_index, test_index) in enumerate(skf.split(df['image_id'].unique(), image_classes)):
        # Get the images for this group
        group_images = df['image_id'].unique()[test_index]
        # Get the rows for this group
        group_rows = df[df['image_id'].isin(group_images)]
        # Save the group to a file
        group_rows.to_csv(f'Daten/group_{i}.csv', index=False)

        # Create a violin plot for the original data
        plt.figure(figsize=(10, 5))
        sns.violinplot(x="Pixel Count", y="Class", data=df, orient='h', cut=0, inner='quart', density_norm='count')
        plt.title('Original distribution')
        plt.tight_layout()
        plt.savefig(f'Daten/original_distribution.png')

        # Create a violin plot for the k-fold data
        plt.figure(figsize=(10, 5))
        sns.violinplot(x="Pixel Count", y="Class", data=group_rows, orient='h', cut=0, inner='quart', density_norm='count')
        plt.title(f'Distribution for fold {i}')
        plt.tight_layout()
        plt.savefig(f'Daten/distribution_fold_{i}.png')