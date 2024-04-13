from Helper.Helper_functions import *
from Helper.ml_models import *

image_dir = 'KittiDaten/training/image_2'
annotation_dir = 'KittiDaten/training/semantic_rgb'

dataset = CustomDataSet(image_dir=image_dir, annotation_dir=annotation_dir)

_, annotation = dataset.__getitem__(0)

print(annotation.shape)

