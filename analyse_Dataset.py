from Helper.Helper_functions import *
from Helper.ml_models import *
from torchvision.utils import save_image

image_dir = 'KittiDaten/training/image_2'
annotation_dir = 'KittiDaten/training/semantic_rgb'

dataset = CustomDataSet(image_dir=image_dir, annotation_dir=annotation_dir)

_, annotation = dataset.__getitem__(0)

#save_image(annotation, 'Daten/tensor_test.png')



print(annotation.shape)

df, df2 = analyse_dataset_RGB(annotation_dir)

class_distribution_violin_plot(df)
stratified_kfold_and_violin_plot(df2)