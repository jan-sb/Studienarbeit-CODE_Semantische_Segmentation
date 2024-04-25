import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
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
from torch.utils.tensorboard.writer import SummaryWriter
import datetime 


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
        #tensor = self.postprocess(tensor)
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
            (128, 64, 128),  # road, 1
            (244, 35, 232),  # sidewalk, 2
            (70, 70, 70),  # building, 3
            (102, 102, 156),  # wall, 4
            (190, 153, 153),  # fence, 5
            (153, 153, 153),  # pole, 6
            (250, 170, 30),  # traffic light, 7 
            (220, 220, 0),  # traffic sign, 8 
            (107, 142, 35),  # vegetation, 9 
            (152, 251, 152),  # terrain, 10
            (70, 130, 180),  # sky, 11
            (220, 20, 60),  # person, 12
            (255, 0, 0),  # rider, 13
            (0, 0, 142),  # car, 14
            (0, 0, 70),  # truck, 15
            (0, 60, 100),  # bus, 16
            (0, 80, 100),  # train, 17
            (0, 0, 230),  # motorcycle, 18
            (119, 11, 32),  # bicycle, 19
            (0, 0, 0),  # unlabeled, 20
            (255,255,255) # test, 21
        ]
        self.num_classes = len(self.city_label_color_map)
        self.learning_rate = 1*10**(-5)
        
        
        
        
        super().__init__(model_name, weights=None, width=width, height=height, pretrained=True, num_classes = self.num_classes)
        # self.preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     SemanticSegmentation,
        #     transforms.Resize(520),
        # ])
        
        self.preprocess = transforms.Compose([
            transforms.Resize((520,520)),
            transforms.PILToTensor(),
            SemanticSegmentation(resize_size=520),
        ])

        self.prepare_model_training()

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
                self.prepare_model_training()
                self.loss = 0
                self.save_model()
                #latest_file_path = self.model_folder_path + f'_latest_{self.model_name}.pth'
                checkpoint = torch.load(path_to_latest)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
                print(f'Successfully loaded Model')
            except:
                print(f'Failed to initialise new model')
                sys.exit()
                
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'Own_Weights/{weights_name}/runs/{current_time}'
        self.writer = SummaryWriter(log_dir=log_dir)

        self.model.eval()

    def own_model_inference_live_no_grad(self, image):
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
                colors=self.city_label_color_map,
                alpha=0.9
            )
            res_image = self.tensor_to_image(res_image)
            return res_image



    def prepare_model_training(self, dataset_train=None, dataset_val= None, batch_size=3, shuffle=True, learning_rate= 1*10**(-5), momentum=0.9, weight_decay=0.0005):
        if dataset_train is not None:
            self.training_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
            print(f'Training Dataset prepared')
        
        if dataset_val is not None: 
            self.val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
            print(f'Validation Dataset prepared')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

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
        for i in range(self.epoch - 5, 0):
            if i % 5 != 0 and self.epoch >= 10 and file_management == True:  # Check if the epoch is not a milestone epoch
                old_filepath = os.path.join(self.folder_path, f'{self.weights_name}_epoch-{i}_{self.model_name}.pth')
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
            else:
                break

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
                
                loss = self.criterion(outputs, labels)            
                loss.backward()
                
                self.optimizer.step()
                run_loss += loss.item()
                
                # For Tensorboard
                
                #step = epoch * len(self.training_loader) + 1
                #self.writer.add_scalars('Loss', {'Train': loss.item()}, step)
                
            epoch_loss = run_loss / len(self.training_loader)
            print(f'Epoch {epoch + 1} von {epochs}    |   Loss: {epoch_loss}')
            self.writer.add_scalar('Epoch Train Loss', epoch_loss, self.epoch)
        self.writer.flush()
        self.writer.close()
        self.save_model()
        
    def validate(self, val_loader):
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)['out']
                _, labels = labels.max(dim=1)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
            avg_loss = total_loss / len(val_loader)
            print(f'Validation Loss: {avg_loss}')
            self.writer.add_scalars('Loss', {'Validation': avg_loss}, self.epoch)
            return avg_loss
        
    def auto_train(self, epochs, l2_lambda=0.01, deviation_threshold=0.1, max_deviations=5):
        deviations = 0
        for epoch in range(epochs):
            self.model.train()
            run_loss = 0.0
            self.epoch += 1
            print(f'Epoch {epoch + 1} von {epochs}')
            counter = 0
            for images, labels in self.training_loader:
                counter +=1
                print(f'Image {counter} von {len(self.training_loader)}')
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)['out']                    
                _, labels = labels.max(dim=1)
                
                loss = self.criterion(outputs, labels)            
                loss.backward()
                
                self.optimizer.step()
                run_loss += loss.item()
                
                # For Tensorboard
                
                step = epoch * len(self.training_loader) + counter
                self.writer.add_scalar('Training Loss', loss.item(), step)
                
            epoch_loss = run_loss / len(self.training_loader)
            print(f'Epoch {epoch + 1} von {epochs}    |   Loss: {epoch_loss}')
            # Validate the model after each epoch
            val_loss = self.validate(self.val_loader)
            self.writer.add_scalars('Epoch and Validation Loss', {'Epoch Loss': epoch_loss, 'Validation Loss': val_loss}, self.epoch)
            
            if val_loss > epoch_loss + deviation_threshold:
                print(f'Validation loss deviated too much from training loss in epoch {self.epoch + 1}')
                deviations += 1
                if deviations > max_deviations:
                    print(f'Stopped training due to validation loss deviating too much from training loss in epoch {self.epoch + 1}')
                    break
            torch.cuda.empty_cache()
        self.writer.flush()
        self.writer.close()
        self.save_model()

class CustomDataSet(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.mean = (0.2892, 0.3272, 0.2867)
        self.std = (0.1904, 0.1932, 0.1905)
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.preprocess = transforms.Compose([
            transforms.Resize((520,520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.preprocess_annotation = transforms.Compose([
            transforms.Resize((520,520)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.counter = 0

        self.image_files = os.listdir(image_dir)
        self.annotation_files = os.listdir(annotation_dir)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            annotation_name = os.path.join(self.annotation_dir, self.annotation_files[idx])

            image = Image.open(img_name).convert("RGB")            
            annotation = Image.open(annotation_name).convert("L") 
                    
            if self.preprocess:
                image = self.preprocess(image)
                annotation = self.preprocess_annotation(annotation)

            return image, annotation

class K_Fold_Dataset:
    def __init__(self, image_dir, annotation_dir, k_fold_csv_dir, leave_out_fold):
        self.csv_files = [os.path.join(k_fold_csv_dir, file) for file in os.listdir(k_fold_csv_dir) if file.endswith('.csv')]
        self.k_folds = len(self.csv_files)
        
        if leave_out_fold >= self.k_folds:
            raise ValueError("leave_out_fold should be less than the number of folds.")
        
        # Use all but one fold for training and validation
        self.train_val_files = [file for i, file in enumerate(self.csv_files) if i != leave_out_fold]
        self.test_files = [self.csv_files[leave_out_fold]]
        
        # Concatenate all train and validation files into one DataFrame
        train_val_df = pd.concat([pd.read_csv(file) for file in self.train_val_files])
        
        # Apply 80/20 train/validation split
        train_df, val_df = train_test_split(train_val_df, test_size=0.2)
        
        # Convert to list for indexing
        self.train_files = train_df.values.tolist()
        self.val_files = val_df.values.tolist()
        self.test_files = pd.read_csv(self.test_files[0]).values.tolist()

        # Initialize TrainDataset and ValDataset
        self.train_dataset = self.TrainDataset(self.train_files, image_dir, annotation_dir)
        self.val_dataset = self.ValDataset(self.val_files, image_dir, annotation_dir)
        
    class TrainDataset(CustomDataSet):
        def __init__(self, train_files, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            self.image_files = [file[0] for file in train_files]
            self.annotation_files = [file[0] for file in train_files]

    class ValDataset(CustomDataSet):
        def __init__(self, val_files, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            self.image_files = [file[0] for file in val_files]
            self.annotation_files = [file[0] for file in val_files]