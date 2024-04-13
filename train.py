from Helper.ml_models import *
from Helper.Helper_functions import *
import matplotlib.pyplot as plt


test1 = TrainedModel('deeplabv3_resnet101', 1242, 375, 'test2', start_epoch='latest')

image_dir = 'KittiDaten/training/image_2'
annotation_dir = 'KittiDaten/training/semantic'

dataset = CustomDataSet(image_dir=image_dir, annotation_dir=annotation_dir)

# test1.pepare_model_training(dataset=dataset, batch_size=4)

# test1.train(2)

#mean, std = calculate_multi_normalization_values(('KittiDaten/training/image_2', 'CityscapesDaten/images'))


#print(f'Mean: {mean}, Std: {std}')

image, labeled = dataset.__getitem__(1)


plt.imshow(image.permute(1, 2, 0))
plt.savefig('Daten/test_image2.png')

save_tensor_as_png(image, 'Daten/test_image.png')
save_tensor_as_png(labeled, 'Daten/test_labeled.png')

