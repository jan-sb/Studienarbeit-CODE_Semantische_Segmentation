from Helper.ml_models import *
from Helper.Helper_functions import *


path = 'Daten/vid.avi'
def main():
    cap, width, height, length, vid_fps = cap_def(path)
    output = video_writer('Daten',
                          vid_fps,
                          (width, height))



    deeplv3 = TorchModel('fcn_resnet50',
                          'FCN_ResNet50_Weights',
                         width,
                         height,
                          pretrained=True)
    counter = 0
    while True:
        counter +=1
        ret, frame = cap.read()
        if not ret:
            break

        image_result = deeplv3.model_inference_live_no_grad(frame)
        #cv.imshow('Result', image_result)

        output.write(frame)

        console_sting = f'Frame {counter}/{length}'
        bar = update_progress_bar(counter, length, bar_length=20)
        update_console(f'{console_sting} \n {bar}' )


        '''keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break'''







if __name__ == '__main__':
    main()