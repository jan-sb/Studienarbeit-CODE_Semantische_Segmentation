from Helper.Helper_functions import *
from Helper.ml_models import *
from torchvision.utils import save_image

kit_image_dir = 'KittiDaten/training/image_2'
kit_annotation_dir = 'KittiDaten/training/semantic'

cit_image_dir = 'CityscapesDaten/images'
cit_annotation_dir = 'CityscapesDaten/semantic'


df = analyse_dataset_GRAY(kit_annotation_dir)
class_distribution_violin_plot(df, output='KittiDaten')
stratified_kfold_and_violin_plot(df, output='KittiDaten',  k=5)


df2 = analyse_dataset_GRAY(cit_annotation_dir)
class_distribution_violin_plot(df2, output='CityscapesDaten')
stratified_kfold_and_violin_plot(df2, output='CityscapesDaten',  k=5)