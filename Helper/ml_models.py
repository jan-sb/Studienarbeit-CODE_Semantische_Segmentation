import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import *
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader
import datetime


class Model:
    def __init__(self, model_name, weights, width, height, pretrained=True):
        self.model_name = model_name
        self.weights = weights
        self.pretrained = pretrained
        self.orig_dim = (height, width)
        self.postprocess = transforms.Compose([transforms.Resize(self.orig_dim)])
        self.start_epoch = 0
        self.epoch = 0

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using CUDA GPU')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print('Using MPS GPU')
        else:
            self.device = torch.device('cpu')
            print("No GPU available. Running on CPU.")

        if model_name in globals():  # Check if model is callable
            model_funktion = globals()[model_name]
            self.model = model_funktion(weights=self.weights).to(self.device)
            self.model.eval()
            print(f'Model loaded: {self.model_name} | Device: {self.device} ')

        else:
            print(f'Error loading model in class Model with {model_name}')
            sys.exit()

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


class HubModel(Model):  # Doesn't work because calls and returns for different Hub models differ to much
    def __init__(self, repo_dir, model_name, width, height, weights, pretrained=True):
        super().__init__(model_name, weights, width, height, pretrained)

        self.model = torch.hub.load(f'{repo_dir}', model_name, pretrained=self.pretrained)


class TorchModel(Model):
    def __init__(self, model_name, weights, width, height, pretrained=True):
        if weights in globals():
            specific_weight = 'DEFAULT'
            weights_call = globals()[weights]
            weights_function = getattr(weights_call, specific_weight)
            try:
                self.weights = weights_function
            except AttributeError:
                print(f'Error loading weights with {weights} and {specific_weight}')
            try:  # Prepare Preprocessing for Frames
                self.preprocess = transforms.Compose([transforms.ToTensor(),
                                                      self.weights.transforms()])
                print(f'Preprocess established')
            except AttributeError:
                print(f'Error preparing preprocess for images in class torch_model')
            print(f'Weights loaded: {weights}')
        else:
            print(f'Error loading weights in class torch_model with {weights}')
            sys.exit()

        super().__init__(model_name, self.weights, width, height, pretrained)




class TrainedModel(Model):

    def __init__(self, model_name, width, height, weights_name, start_epoch='latest', pretrained=True):
        super().__init__(model_name, weights=None, width=width, height=height, pretrained=True)
        self.num_classes = len(self.label_color_map)

        # Make Folder for the Trained Weights
        self.folder_path = 'Own_Models'
        self.weights_name = weights_name
        self.model_folder_path = os.path.join(self.folder_path, self.weights_name)

        self.model[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))


        # Loading the Model
        search_file_path = os.path.join(self.folder_path, f'{self.weights_name}')
        if os.path.exists(search_file_path) and start_epoch == 'latest':
            try:
                latest_file_path = search_file_path + '_latest_{self.model_name}.pth'
                checkpoint = torch.load(latest_file_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
            except:
                print(f'Error loading Model with Epoch latest in Class TrainedModel')
                sys.exit()
        elif os.path.exists(search_file_path) and start_epoch != 'latest':
            try:
                epoch_file_path = search_file_path + f'_epoch-{self.epoch}_{self.model_name}'
                checkpoint = torch.load(epoch_file_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
            except:
                print(f'Error loading Model with Epoch {start_epoch} in Class TrainedModel')
                sys.exit()
        elif os.path.exists(search_file_path) != True:
                print(f'Model file \'Own_Weights\' doesnt exist')
                sys.exit()
        else:
            print(f'Latest Epoch Save doesnt exist or Epoch Number Save doesnt exist, initialising new Save')
            try:
                self.save_model()
                latest_file_path = search_file_path + '_latest_{self.model_name}.pth'
                checkpoint = torch.load(latest_file_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
                print(f'Successfully loaded Model')
            except:
                print(f'Failed to initialise new model')
                sys.exit()

        self.model.eval()


    def pepare_model_training(self, dataset, batch_size=10, shuffle=True, learning_rate = 1*10**(-5)):
        self.training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)



    def save_model(self):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        weights_name_current = f'{self.model_folder_path}_epoch-{self.epoch}_{self.model_name}'
        weights_name_latest = f'{self.model_folder_path}_latest_{self.model_name}'

        torch.save({
            'epoch':self.epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'loss':self.loss,
        }, weights_name_current)
        torch.save({
            'epoch':self.epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'loss':self.loss,
        }, weights_name_latest)

        # Delete saved models from last five epochs, except milestone epochs
        for i in range(self.epoch - 5, 0) and self.epoch >= 10:
            if i % 5 != 0:  # Check if the epoch is not a milestone epoch
                old_filepath = os.path.join(self.folder_path, f'{self.weights_name}_epoch-{i}_{self.model_name}.pth')
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            run_loss = 0.0
            for images, labels in self.training_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                run_loss += loss.item()
            epoch_loss = run_loss / len(self.training_loader)
            print(f'Epoch {epoch+1} von {epochs}    |   Loss: {epoch_loss}')

