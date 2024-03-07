from Helper.ml_models import *
from Helper.Helper_functions import *


path = 'Daten/vid.avi'
def main():
    cap, width, height, length, fps = cap_def(path)


    deeplv3 = TorchModel('fcn_resnet50',
                          'FCN_ResNet50_Weights',
                         width,
                         height,
                          pretrained=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_result = deeplv3.model_inference_live_no_grad(frame)
        cv.imshow('Result', image_result)


        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break







if __name__ == '__main__':
    main()