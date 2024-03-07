import cv2 as cv
import numpy as np
from Helper.canny import canny_edge_detection
from Helper.Helper_functions import *
import datetime
from Tests.test3 import *



def main():

    path = "Daten/vid.avi"

    #test_instance = model_test(model_s='deeplabv3_resnet50', weights='DeepLabV3_ResNet50_Weights.DEFAULT', path='Daten/vid.avi', live='write', output_path='Ergebnisse')
    #test_instance = model_test(model_s='deeplabv3_resnet50', weights='DeepLabV3_ResNet50_Weights.DEFAULT',path='Daten/vid.avi', live='True')
    #test_instance.prep_model_infer()
    #test_instance.model_inference()

    cap = cv.VideoCapture(path)
    k = 0
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    prev_markers, prev_colors = None, {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading video")
            break

        # improve image
        kernelsize =11
        gauss = cv.GaussianBlur(frame, (kernelsize, kernelsize), cv.BORDER_DEFAULT)
        highfreq = frame - gauss
        frame = frame + highfreq * 3

        erg_frame, markers = watershed_segmentation(frame)

        # Apply watershed algorithm to obtain segmented image

        #erg_frame = cv.addWeighted(frame, 0.9, markers, 0.9, gamma=0)
        # Frame Ausgabe
        cv.imshow("watershed", erg_frame)
        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break
        print(f'Frame {k}/{length}')
        k +=1







if __name__ == "__main__":
    main()
