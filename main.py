from Helper.ml_models import *
from Helper.Helper_functions import *
from cityscapesscripts.helpers.labels import labels

path = 'Input_Daten/vid.avi'
def main():
    cap, width, height, length, vid_fps = cap_def(path)
    output = video_writer('Daten',
                          vid_fps,
                          (width, height))#
    
    output2 = video_writer('Daten2',
                           vid_fps,
                          (width, height))#
                           

    # deeplabv3_mobilenet_v3_large | DeepLabV3_MobileNet_V3_Large_Weights
    # fcn_resnet50 | FCN_ResNet50_Weights

    # deeplv3 = TorchModel('deeplabv3_mobilenet_v3_large',
    #                       'DeepLabV3_MobileNet_V3_Large_Weights',
    #                      width,
    #                      height,
    #                       pretrained=True)
    deeplv3 = TorchModel('fcn_resnet50',
                          'FCN_ResNet50_Weights',
                         width,
                         height,
                          pretrained=True)
    
    counter = 0
    while counter < 300:
        counter +=1
        ret, frame = cap.read()
        if not ret:
            break

        image_result = deeplv3.model_inference_live_no_grad(frame)
        #print(f'IMAGE RESULT: {image_result.shape}')
        image_result = cv.addWeighted(frame, 0.9, image_result, 0.9, gamma=0)
        output.write(image_result)
        output2.write(image_result)

        console_sting = f'Frame {counter}/{length}'
        bar = update_progress_bar(counter, length, bar_length=20)
        update_console(f'{console_sting} \n {bar}' )


        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break


    output.release()
    cap.release()


if __name__ == '__main__':
    main()