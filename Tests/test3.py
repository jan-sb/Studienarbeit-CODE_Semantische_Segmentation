import torch
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import cv2 as cv
import numpy as np
from torchvision.utils import make_grid
from torchvision.io import read_image
from PIL import Image

class model_test:
    def __init__(self, model_s:str='fcn_resnet50', weights:str='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1', path:str='' ,live:str='False', output_path:str=''):

        #self.model = globals().get(model_s)
        #self.weights = globals().get(weights)
        #self.weights = FCN_ResNet50_Weights.DEFAULT


        self.model = fcn_resnet50
        self.weights = FCN_ResNet50_Weights.DEFAULT

        if self.model and callable(self.model):
            s_model = 'model loaded'
        else: s_model = 'model loading error'

        self.device = torch.device('mps')

        if self.weights and callable(self.weights):
            s_weights = 'weights loaded'
        else: s_weights = 'weights loading error'

        if path:
            self.path = path
            s_path = 'path provided'
        else: s_path = 'no path provided' # change for raise Exception
        self.live = live
        self.output_path = output_path

        self.label_color_map = label_color_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

        print(f'Init: {s_model} | {s_weights} | {s_path}')


    def prep_model_infer(self):
        self.model = self.model(weights= self.weights)
        self.model.eval()
        print(self.weights)
        self.preprocess = self.weights.transforms()

        print('Model prepared for inference')
        return None

    def image_to_prepro_tensor(self, image):
        preprocess = transforms.Compose([transforms.ToTensor(), self.preprocess])
        tensor = preprocess(image).unsqueeze(0)
        self.orig_dim = image.shape[0:2]
        print('Preprocessed Tensor successful')
        return tensor

    def tensor_to_image(self, tensor):
        print('Original Dimensions', self.orig_dim)
        postprocess = transforms.Compose([
            transforms.Resize(self.orig_dim)
        ])
        tensor = postprocess(tensor)
        image = np.array(tensor, dtype='uint8')
        image = np.transpose(image, (1, 2, 0))
        return image

    def model_inference(self):
        cap = cv.VideoCapture(self.path)
        # if images come from opencv, they need to be transformed to tensors, np.ndarrays are not an accepted input of the preprocess call
        preprocess = transforms.Compose([transforms.ToTensor(),self.preprocess])


        if self.output_path:
            print(f'Output in {self.output_path}/output.mp4')
            w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
            out = cv.VideoWriter(self.output_path+'/output.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        k = 0
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame = cap.read()
            print('Original Frame Shape ', frame.shape)
            if not ret:
                print('Video file empty or process finished')
                break

            im0 = self.image_to_prepro_tensor(frame)
            im1 = torch.zeros(torch.squeeze(im0).size(), dtype=torch.uint8)

            with torch.no_grad():
                # Keys for model are 'aux' and 'out'
                output = self.model(im0)['out']

            num_classes = output.shape[1]
            masks = output
            all_masks = masks.argmax(1) == torch.arange(num_classes)[:, None, None]


            if self.live=='True':
                seg_result = draw_segmentation_masks(
                    im1,
                    all_masks,
                    colors=self.label_color_map,
                    alpha=0.9
                )

                result_image = self.tensor_to_image(seg_result)
                #print(f'Result Image Type {type(result_image)} | Result Image Dtype: {result_image.dtype}| Result Image Shape: {result_image.shape} | Frame Type: {type(frame)} | Frame Image Dtype: {frame.dtype} | Frame Shape: {frame.shape}')

                ergebnis_frame = cv.addWeighted(result_image, 0.9, frame, 0.8, gamma= 0)

                cv.imshow('Result', ergebnis_frame)

                keyboard = cv.waitKey(1)
                if keyboard == 'q' or keyboard == 27:
                    break
            elif self.live == 'write':
                seg_result = draw_segmentation_masks(
                    im1,
                    all_masks,
                    colors=self.label_color_map,
                    alpha=0.9
                )
                result_image = self.tensor_to_image(seg_result)
                ergebnis_frame = cv.addWeighted(result_image, 0.9, frame, 0.8, gamma=0)
                out.write(ergebnis_frame)
                print(f'Frame {k}/{length}')
                k +=1
            elif self.live == 'false':
                None
            else:
                print('Fehlerhafter Live String')
                break




        cap.release()
        cv.destroyAllWindows()