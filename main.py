import cv2 as cv
import numpy as np
from Helper.canny import canny_edge_detection
import datetime
from Tests.test3 import *



def main():

    path = ""
    #pytorch_models(path)

    test_instance = model_test(model_s='deeplabv3_resnet50', weights='DeepLabV3_ResNet50_Weights.DEFAULT', path='Daten/vid.avi', live='write', output_path='Ergebnisse')
    test_instance.prep_model_infer()
    test_instance.model_inference()



if __name__ == "__main__":
    main()
