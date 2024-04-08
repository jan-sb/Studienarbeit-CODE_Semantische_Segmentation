import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import *
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms._presets import SemanticSegmentation
from functools import partial
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class Model:
    def __init__(self, model_name, weights, width, height, pretrained=True, num_classes = 21):
        self.model_name = model_name
        self.weights = weights
        self.pretrained = pretrained
        self.orig_dim = (height, width)
        self.postprocess = transforms.Compose([transforms.Resize(self.orig_dim)])
        self.start_epoch = 0
        self.epoch = 0
        self.num_classes = num_classes

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
            self.model = model_funktion(weights=self.weights, num_classes=self.num_classes).to(self.device)
            self.model.eval()
            print(f'Model loaded: {self.model_name} | Device: {self.device} ')

        else:
            print(f'Error loading model in class Model with {model_name}')
            sys.exit()

        self.old_label_color_map = [
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

        self.city_label_color_map = [
            (128, 64, 128),  # road,
            (244, 35, 232),  # sidewalk,
            (70, 70, 70),  # building,
            (102, 102, 156),  # wall,
            (190, 153, 153),  # fence,
            (153, 153, 153),  # pole,
            (250, 170, 30),  # traffic light,
            (220, 220, 0),  # traffic sign,
            (107, 142, 35),  # vegetation,
            (152, 251, 152),  # terrain,
            (70, 130, 180),  # sky,
            (220, 20, 60),  # person,
            (255, 0, 0),  # rider,
            (0, 0, 142),  # car,
            (0, 0, 70),  # truck,
            (0, 60, 100),  # bus,
            (0, 80, 100),  # train,
            (0, 0, 230),  # motorcycle,
            (119, 11, 32),  # bicycle,
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
            print(output.shape)
            all_masks = output.argmax(1) == torch.arange(num_classes, device=self.device)[:, None, None]
            tensor = tensor.to(torch.uint8).squeeze(0)
            res_image = draw_segmentation_masks(
                tensor,
                all_masks,
                colors=self.old_label_color_map,
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
        self.city_label_color_map = [
            (128, 64, 128),  # road,
            (244, 35, 232),  # sidewalk,
            (70, 70, 70),  # building,
            (102, 102, 156),  # wall,
            (190, 153, 153),  # fence,
            (153, 153, 153),  # pole,
            (250, 170, 30),  # traffic light,
            (220, 220, 0),  # traffic sign,
            (107, 142, 35),  # vegetation,
            (152, 251, 152),  # terrain,
            (70, 130, 180),  # sky,
            (220, 20, 60),  # person,
            (255, 0, 0),  # rider,
            (0, 0, 142),  # car,
            (0, 0, 70),  # truck,
            (0, 60, 100),  # bus,
            (0, 80, 100),  # train,
            (0, 0, 230),  # motorcycle,
            (119, 11, 32),  # bicycle,
        ]
        self.num_classes = len(self.city_label_color_map)
        self.learning_rate = 1*10**(-5)

        super().__init__(model_name, weights=None, width=width, height=height, pretrained=True, num_classes = self.num_classes)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            SemanticSegmentation,
            transforms.Resize(520),
        ])

        self.pepare_model_training()

        # Make Folder for the Trained Weights
        self.folder_path = 'Own_Weights'
        self.weights_name = weights_name
        self.model_folder_path = os.path.join(self.folder_path, self.weights_name)

        #self.model[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))

        # Loading the Model
        path_to_latest = self.model_folder_path+ f'/{self.weights_name}_latest_{self.model_name}.pth'
        path_to_epoch = self.model_folder_path+ f'/{self.weights_name}_epoch-{self.epoch}_{self.model_name}.pth'
        if os.path.exists(path_to_latest) and start_epoch == 'latest':
            try:
                checkpoint = torch.load(path_to_latest)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
            except:
                print(f'Error loading Model with Epoch latest in Class TrainedModel')
                sys.exit()
        elif os.path.exists(path_to_epoch) and start_epoch != 'latest':
            try:
                checkpoint = torch.load(path_to_epoch)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
            except:
                print(f'Error loading Model with Epoch {start_epoch} in Class TrainedModel')
                sys.exit()
        elif os.path.exists(self.model_folder_path) != True:
            print(f'Model directory {self.model_folder_path} doesnt exist')
            sys.exit()
        else:
            print(f'Latest Epoch Save doesnt exist or Epoch Number Save doesnt exist, initialising new Save')
            try:
                self.pepare_model_training()
                self.loss = 0
                self.save_model()
                latest_file_path = self.model_folder_path + '_latest_{self.model_name}.pth'
                checkpoint = torch.load(latest_file_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
                print(f'Successfully loaded Model')
            except:
                print(f'Failed to initialise new model')
                sys.exit()
                
        self.writer = SummaryWriter(log_dir='Own_Weights/Logs')

        self.model.eval()



    def pepare_model_training(self, dataset=None, batch_size=3, shuffle=True, learning_rate= 1*10**(-5)):
        if dataset is not None:
            self.training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            print(f'Training Loader prepared')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def save_model(self, file_management=False):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        weights_name_current = f'{self.model_folder_path}/{self.weights_name}_epoch-{self.epoch}_{self.model_name}.pth'
        weights_name_latest = f'{self.model_folder_path}/{self.weights_name}_latest_{self.model_name}.pth'

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, weights_name_current)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, weights_name_latest)

        # Delete saved models from last five epochs, except milestone epochs
        for i in range(self.epoch - 5, 0) and self.epoch >= 10 and file_management == True:
            if i % 5 != 0:  # Check if the epoch is not a milestone epoch
                old_filepath = os.path.join(self.folder_path, f'{self.weights_name}_epoch-{i}_{self.model_name}.pth')
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)

        print(f'Saved Model')

    def train(self, epochs, l2_lambda=0.01):
        self.model.train()
        torch.cuda.empty_cache()
        for epoch in range(epochs):
            run_loss = 0.0
            self.epoch += 1
            for images, labels in self.training_loader:
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)['out']                    
                _, labels = labels.max(dim=1)
                
                ce_loss = F.cross_entropy(outputs, labels)                
                # Calculate L2 regularization loss
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, p=2)                
                # Total loss is the sum of cross-entropy loss and L2 regularization loss
                loss = ce_loss + l2_lambda * l2_reg
                
                
                loss.backward()
                self.optimizer.step()
                run_loss += loss.item()
                
                # For Tensorboard
                
                step = epoch * len(self.training_loader) + 1
                self.writer.add_scalar('Training Loss', loss.item(), step)
                
            epoch_loss = run_loss / len(self.training_loader)
            print(f'Epoch {epoch + 1} von {epochs}    |   Loss: {epoch_loss}')
            self.writer.add_scalar('Epoch Loss', epoch_loss, epoch)
        self.writer.close()
        self.save_model()


## Kann wahrscheinlich weg
'''    def prepare_dataset(self, path_images, path_labeled):
        if not (os.path.exists(path_images) and os.path.exists(path_labeled)):
            raise FileNotFoundError("One or both of the specified paths do not exist.")

        image_files = os.listdir(path_images)
        for img_file in image_files:
            img_path = os.path.join(path_images, img_file)
            label_file = img_file.replace('.png','_label.png')
            label_path = os.path.join(path_labeled, label_file)

            if not os.path.exists(label_path):
                print(f"For label file {label_file} exists no {img_file}. Skipping...")
                continue

            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')

            image = self.transforms(image)
            label = self.transforms(label)

            self.dataset.append'''


class CustomDataSet(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.preprocess = transforms.Compose([
            transforms.Resize((520,520)),
            transforms.PILToTensor(),
            SemanticSegmentation(resize_size=520),
        ])
        self.counter = 0

        self.image_files = os.listdir(image_dir)
        self.annotation_files = os.listdir(annotation_dir)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            annotation_name = os.path.join(self.annotation_dir, self.annotation_files[idx])
            self.counter +=1
            print(self.counter)

            image = Image.open(img_name).convert("RGB")
            annotation = Image.open(annotation_name).convert("RGB")  # Convert to grayscale

            if self.preprocess:
                image = self.preprocess(image)
                annotation = self.preprocess(annotation)

            return image, annotation
