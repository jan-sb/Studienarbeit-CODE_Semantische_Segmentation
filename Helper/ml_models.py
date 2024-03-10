import sys
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import *
from torchvision.utils import draw_segmentation_masks


class Model:
    def __init__(self, model_name, weights, width, height, pretrained=True):
        self.model_name = model_name
        self.weights = weights
        self.pretrained = pretrained
        self.orig_dim = (width, height)
        self.postprocess = transforms.Compose([transforms.Resize(self.orig_dim)])

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using CUDA GPU')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print('Using MPS GPU')
        else:
            self.device = torch.device('cpu')
            print("No GPU available. Running on CPU.")

        self.label_color_map = [
            (0, 0, 0),  # background
            (128, 0, 0),  # aeroplane
            (0, 128, 0),  # bicycle
            (128, 128, 0),  # bird
            (0, 0, 128),  # boat
            (128, 0, 128),  # bottle
            (0, 128, 128),  # bus
            (128, 128, 128),  # car
            (64, 0, 0),  # cat
            (192, 0, 0),  # chair
            (64, 128, 0),  # cow
            (192, 128, 0),  # dining table
            (64, 0, 128),  # dog
            (192, 0, 128),  # horse
            (64, 128, 128),  # motorbike
            (192, 128, 128),  # person
            (0, 64, 0),  # potted plant
            (128, 64, 0),  # sheep
            (0, 192, 0),  # sofa
            (128, 192, 0),  # train
            (0, 64, 128)  # tv/monitor
        ]

    def image_preprocess(self, image):
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def tensor_to_image(self, tensor):
        tensor = self.postprocess(tensor)
        image = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
        image = np.array(image, dtype='uint8')
        return image

    def model_inference_no_grad(self, image):
        with torch.no_grad():
            tensor = self.image_preprocess(image)
            output = self.model(tensor)['out'].to(self.device)
            return output

    def model_inference_live_no_grad(self, image):
        with torch.no_grad():
            tensor = self.image_preprocess(image)
            output = self.model(tensor)['out'].to(self.device)
            num_classes = output.shape[1]
            all_masks = output.argmax(1) == torch.arange(num_classes, device=self.device)[:, None, None]
            tensor = tensor.to(torch.uint8).squeeze(0)
            res_image = draw_segmentation_masks(
                tensor,
                all_masks,
                colors=self.label_color_map,
                alpha=0.9
            )
            res_image = self.tensor_to_image(res_image)
            return res_image

class HubModel(Model):  # Doesn't work ecause calls and returns for different Hub models differ to much
    def __init__(self, repo_dir, model_name, width, height, weights, pretrained=True):
        super().__init__(model_name, weights, width, height, pretrained)

        self.model = torch.hub.load(f'{repo_dir}', model_name, pretrained=self.pretrained)


class TorchModel(Model):
    def __init__(self, model_name, weights, width, height, pretrained=True):
        super().__init__(model_name, weights, width, height, pretrained)

        if model_name in globals():  # Check if model is callable
            model_funktion = globals()[model_name]
            if weights in globals():
                specific_weight = 'DEFAULT'
                weights_call = globals()[weights]
                weights_function = getattr(weights_call, specific_weight)
                try:
                    weights = weights_function
                except AttributeError:
                    print(f'Error loading weights with {weights} and {specific_weight}')
                try:  # Prepare Preprocessing for Frames
                    self.preprocess = transforms.Compose([transforms.ToTensor(),
                                                          weights.transforms()])
                except AttributeError:
                    print(f'Error preparing preprocess for images in class torch_model')
                print(f'Weights loaded: {weights}')
            else:
                print(f'Error loading weights in class torch_model with {weights}')
                sys.exit()
            self.model = model_funktion(weights=weights).to(self.device)
            self.model.eval()
            print(f'Model loaded: {self.model_name} | Device: {self.device} ')

        else:
            print(f'Error loading model in class torch_model with {model_name}')
            sys.exit()





