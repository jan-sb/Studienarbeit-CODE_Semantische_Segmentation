import sys

import torch
from torchvision.models.segmentation import *
from torchvision import transforms
import Helper_functions
import numpy as np



class model():
    def __init__(self, model_name, weights, width, height,  pretrained=True):
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






class hub_model(model):
    def __init__(self,repo_dir, model_name, weights, pretrained=True):
        super().__init__(model_name, weights, pretrained)

        self.model = torch.hub.load(f'{repo_dir}' , model_name, pretrained=self.pretrained)
        



class torch_model(model):
    def __init__(self, model_name, weights, pretrained=True):
        super().__init__(model_name, weights, pretrained)

        if model_name in globals(): # Check if model is callable
            model_funktion = globals()[model_name]
            if weights in globals():
                specific_weight = 'DEFAULT'
                weights_function = getattr(FCN_ResNet50_Weights, specific_weight)
                try:
                    weights = weights_function
                except AttributeError:
                    print(f'Error loading weights with {weights} and {specific_weight}')
                try:    # Prepare Preprocessing for Frames
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


    def image_preprocess(self, image):
        return self.preprocess(image).unsqueeze(0).to(self.device)


    def tensor_to_image(self, tensor, width, height):
        tensor = self.postprocess(tensor)
        image = np.transpose(tensor.cpu().numpy(), (1, 2, 0))







