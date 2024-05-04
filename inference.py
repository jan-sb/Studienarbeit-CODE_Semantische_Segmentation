from Helper.ml_models import *
from Helper.Helper_functions import *
from PIL import Image
import cv2 as cv


model = TrainedModel('deeplabv3_resnet101', 2048, 1024, 'deeplabv3_resnet101_k_fold_0', start_epoch='latest')
model2 = TorchModel('deeplabv3_resnet101', 'DeepLabV3_ResNet101_Weights', 2048, 1024, pretrained=True)

output_path = 'Daten2'

path = 'CityscapesDaten/images'
image = Image.open(path + '/000000_01.png')


inf_result1 = model.own_model_inference_live_no_grad(image)
inf_result2 = model2.model_inference_live_no_grad(image)
cv.imwrite(output_path + '/test1.png', inf_result1)
cv.imwrite(output_path + '/test2.png', inf_result2)









