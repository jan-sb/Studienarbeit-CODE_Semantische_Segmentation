from __future__ import print_function, absolute_import, division
from collections import namedtuple
from cityscapesscripts.helpers.labels import *
import Helper.Helper_functions as hf



print("List of cityscapes labels:")
print("")
print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12} | {:>14}".format('name', 'id', 'trainId', 'category',
                                                                              'categoryId', 'hasInstances',
                                                                              'color', 'ignoreInEval'))
print("    " + ('-' * 98))
counter = 0
for label in labels:
    #if label.ignoreInEval ==0:
    counter +=1
    print("     {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12} | {}".format(label.name, label.id, label.trainId,
                                                                                  label.category, label.categoryId,
                                                                                  label.hasInstances,
                                                                                  str(label.color),  # Convert the tuple to a string
                                                                                  label.ignoreInEval,
                                                                                  ))
print(" ", '\n', counter)

label_dict = {}
for label in labels:
    if label.trainId != -1:
        if label.trainId not in label_dict:
            label_dict[label.trainId] = []
            label_dict[label.trainId].append([label.id, label.name, str(label.color)])
        else:
            label_dict[label.trainId].append([label.id, label.name, str(label.color)]) 
        #print(f'{label.id}, #{label.name}, Nr: {counter}')


hf.create_ground_truth_json()

hf.create_ground_truth('CityscapesDaten/semantic_default', 'CityscapesDaten/semantic')

# print("Example usages:")

# # Map from name to label
# name = 'car'
# id = name2label[name].id
# print("ID of label '{name}': {id}".format(name=name, id=id))

# # Map from ID to label
# category = id2label[id].category
# print("Category of label with ID '{id}': {category}".format(id=id, category=category))

# # Map from trainID to label
# trainId = 0
# name = trainId2label[trainId].name
# print("Name of label with trainID '{id}': {name}".format(id=trainId, name=name))









# import os
# import cv2
# import numpy as np

# # Define the directory path
# dir_path = 'CityscapesDaten/semantic'
# image_dir_path = 'CityscapesDaten/images'

# # Define the integers to look for
# a= 17
# b = 18
# integers_to_find = {a,b}

# # Initialize the counter outside the loop
# counter = 0

# # Loop over all files in the directory
# for filename in os.listdir(dir_path):
#     # Construct the full file path
#     file_path = os.path.join(dir_path, filename)
    
#     # Load the image in grayscale mode
#     image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
#     # If the image was loaded successfully, check if it contains the integers
#     if image is not None:
#         unique_values = np.unique(image)
#         if integers_to_find.intersection(unique_values):
#             print(f"Image {file_path} contains one of the integers {integers_to_find}.")
            
#             # Make all values 0 except for a and b
#             image[(image != a) & (image != b)] = 0
            
#             # Create a color mask for each integer
#             mask_a = np.zeros_like(image)
#             mask_b = np.zeros_like(image)
#             mask_a[image == a] = 125
#             mask_b[image == b] = 255
            
#             # Combine the masks
#             image = cv2.merge((mask_b, mask_a, np.zeros_like(image)))
            
#             # Load the corresponding image from the 'images' directory
#             image_file_path = os.path.join(image_dir_path, filename)
#             image2 = cv2.imread(image_file_path)
            
#             # If the image was loaded successfully, overlay the images
#             if image2 is not None:
#                 # Resize the semantic image to match the size of the original image
#                 image = cv2.resize(image, (image2.shape[1], image2.shape[0]), interpolation = cv2.INTER_NEAREST)
                
#                 # Overlay the images with a 50/50 weight
#                 overlay = cv2.addWeighted(image2, 0.5, image, 0.5, 0)
                
#                 # Save the image
#                 cv2.imwrite(f'Daten2/overlayed/output_{counter}.png', overlay)
                
#                 # Increment the counter
#                 counter += 1
                
#                 # If 10 images have been saved, break the loop
#                 if counter >= 10:
#                     break