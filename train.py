from Helper.ml_models import *


test1 = TrainedModel('deeplabv3_resnet101', 1242, 375, 'test1', start_epoch='latest')

image_dir = 'KittiDaten/training/image_2'
annotation_dir = 'KittiDaten/training/semantic'

dataset = CustomDataSet(image_dir=image_dir, annotation_dir=annotation_dir)

test1.pepare_model_training(dataset=dataset)

test1.train(20)

