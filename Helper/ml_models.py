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
from torchvision.utils import draw_segmentation_masks, save_image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms._presets import SemanticSegmentation
from torchvision.transforms import functional as TF
from functools import partial
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime 
import random
from tqdm import tqdm
from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomCrop, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torchvision.utils import make_grid
from torch.cuda.amp import autocast
import json



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
            try:
                self.model = model_funktion(weights=self.weights, num_classes=self.num_classes).to(self.device)
            except FileNotFoundError as e:
                print("[DEBUG] model_funktion call caused:", e)
                raise e 
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
    def __init__(
        self,
        model_name,
        width,
        height,
        weights_name='',
        folder_path="Own_Weights",
        start_epoch="latest",
        pretrained=True,
        writer=None,
        mapillary = False, 
        num_classes = 21,
        skip_local_load=False  # <--- NEW PARAMETER
        
    ):
        
        if  not mapillary:
            self.city_label_color_map = [
                (128, 64, 128),  # ID__0, road
                (244, 35, 232),  # ID__1, sidewalk
                (70, 70, 70),    # ID__2, building
                (102, 102, 156), # ID__3, wall
                (190, 153, 153), # ID__4, fence
                (153, 153, 153), # ID__5, pole
                (250, 170, 30),  # ID__6, traffic light
                (220, 220, 0),   # ID__7, traffic sign
                (107, 142, 35),  # ID__8, vegetation
                (152, 251, 152), # ID__9, terrain
                (70, 130, 180),  # ID__10, sky
                (220, 20, 60),   # ID__11, person
                (255, 0, 0),     # ID__12, rider
                (0, 0, 142),     # ID__13, car
                (0, 0, 70),      # ID__14, truck
                (0, 60, 100),    # ID__15, bus
                (0, 80, 100),    # ID__16, train
                (0, 0, 230),     # ID__17, motorcycle
                (119, 11, 32),   # ID__18, bicycle
                (0, 0, 0),       # ID__19, unlabeled
            ]
            self.num_classes = len(self.city_label_color_map)
        if mapillary: 
            self.num_classes = num_classes
        
        self.step = 0
        self.learning_rate = 1e-5
        if writer is not None:
            self.writer = writer

        # Call the parent class Model's init
        super().__init__(
            model_name,
            weights=None,
            width=width,
            height=height,
            pretrained=pretrained,
            num_classes=self.num_classes
        )

        # Example transform pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.PILToTensor(),
            SemanticSegmentation(resize_size=520),
        ])

        # Prepare for training (will build the optimizer, scheduler, etc.)
        self.prepare_model_training()

        self.val_loss = 0
        self.old_val_loss = self.val_loss

        # Manage folder paths
        self.folder_path = folder_path
        self.weights_name = weights_name
        self.model_folder_path = os.path.join(self.folder_path, self.weights_name)
        
        # Create the folder if it doesn't exist (avoid error):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path, exist_ok=True)
        
        # If we do want to load from a local .pth (and not from a Ray checkpoint),
        # we'll do the usual logic. Otherwise, we skip:
        if not skip_local_load:
            path_to_latest = os.path.join(
                self.model_folder_path, f"{self.weights_name}_latest_{self.model_name}.pth"
            )
            path_to_epoch = os.path.join(
                self.model_folder_path, f"{self.weights_name}_epoch-{start_epoch}_{self.model_name}.pth"
            )

            if os.path.exists(path_to_latest) and start_epoch == "latest":
                try:
                    checkpoint = torch.load(path_to_latest)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.epoch = checkpoint["epoch"]
                    self.val_loss = checkpoint["val_loss"]
                    self.step = checkpoint["step"]
                except Exception as e:
                    print(f"Error loading Model with Epoch latest: {e}")
                    # Instead of sys.exit(), let's just warn
                    print("Skipping local .pth load due to error above.")
            elif os.path.exists(path_to_epoch) and start_epoch != "latest":
                try:
                    checkpoint = torch.load(path_to_epoch)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.epoch = checkpoint["epoch"]
                    self.val_loss = checkpoint["val_loss"]
                    self.step = checkpoint["step"]
                except Exception as e:
                    print(f"Error loading Model with Epoch {start_epoch}: {e}")
                    print("Skipping local .pth load due to error above.")
            else:
                # If no .pth found, we can initialize a new one or skip
                print("No local .pth found; initializing a new model save.")
                try:
                    self.val_loss = 0
                    self.save_model(file_management=False)
                    # Reload from the newly saved model
                    if os.path.exists(path_to_latest):
                        new_checkpoint = torch.load(path_to_latest)
                        self.model.load_state_dict(new_checkpoint["model_state_dict"])
                        self.optimizer.load_state_dict(new_checkpoint["optimizer_state_dict"])
                        self.epoch = new_checkpoint["epoch"]
                        self.val_loss = new_checkpoint["val_loss"]
                        self.step = new_checkpoint["step"]
                        print("Successfully loaded a fresh model checkpoint.")
                    else:
                        print("Failed to find newly saved .pth, continuing anyway.")
                except Exception as e:
                    print(f"Failed to initialise new model: {e}")
        else:
            # If skip_local_load == True, do nothing special
            print("Skipping local .pth load logic (likely using external Ray checkpoint).")
            self.epoch = 0  # or set any default you prefer

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(folder_path, weights_name, "runs", current_time)
        
        self.old_val_loss = self.val_loss
        self.model.eval()
        
        
        
    def inference(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            image = image.unsqueeze(0) 
            output = self.model(image)['out'].to(self.device)
        return output

    def own_model_inference_live_no_grad(self, image):
        self.model.eval()
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



    def prepare_model_training(self, dataset_train=None, dataset_val= None, dataset_test=None, batch_size=3, val_batch_size = 1,shuffle=True, learning_rate= 1*10**(-5), ray_tune = False, weight_decay=0.001, num_workers = 0, pin_memory = False):
        if dataset_train is not None:
            self.training_loader = DataLoader(dataset_train,
                                              batch_size=batch_size,
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                            )
            print(f'Training Dataset prepared')
        
        if dataset_val is not None: 
            self.val_loader = DataLoader(dataset_val,
                                         batch_size=val_batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory,
                                        )
            print(f'Validation Dataset prepared')
        if dataset_test is not None: 
            self.test_loader = DataLoader(dataset_test,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        )
            print(f'Test Dataset prepared')
            
        # if ray_tune == False:             # Redundant with implmentation of own learning rate scheduler
        #     if self.epoch >=15 and self.epoch < 30: 
        #         self.learning_rate = 1*10**(-5)
        #     elif self.epoch >= 30 and self.epoch < 45:
        #         self.learning_rate = 1*10**(-6)
        #     elif self.epoch >= 45 and self.epoch < 55:
        #         self.learning_rate = 1*10**(-7)
        #     elif self.epoch >= 55:
        #         self.learning_rate = 1*10**(-8)
        #     print(f'own lrs: {self.learning_rate}')
        

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        
        

    def save_model(self, file_management=False, save_path = None):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
            
            
        if save_path is None:
            if not os.path.exists(self.model_folder_path):
                os.makedirs(self.model_folder_path)
            weights_name_latest = f'{self.model_folder_path}/{self.weights_name}_latest_{self.model_name}.pth'
        else:
            # Hier nutzt du den Pfad von Ray Tune
            weights_name_latest = os.path.join(save_path, "checkpoint.pth")    
            
        

        weights_name_current = f'{self.model_folder_path}/{self.weights_name}_epoch-{self.epoch}_{self.model_name}.pth'
        weights_name_latest = f'{self.model_folder_path}/{self.weights_name}_latest_{self.model_name}.pth'

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.val_loss,
            'step': self.step,
        }, weights_name_latest)

        # Delete saved models from last five epochs, except milestone epochs
        if (self.val_loss < self.old_val_loss) and (file_management == True):
            for file in os.listdir(self.model_folder_path):
                if file.endswith('.pth') and file != f'{self.weights_name}_latest_{self.model_name}.pth':
                    os.remove(os.path.join(self.model_folder_path, file))
                    print(f'Removed {file}')
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': self.val_loss,
                'step': self.step,
            }, weights_name_current)
            
            self.old_val_loss = self.val_loss
        print(f'Saved Model')

    def train(self, use_autocast):    
        self.model.train()
        run_loss = 0.0
        self.epoch += 1
        correct = 0 
        total = 0 
        if use_autocast: 
            scaler = torch.cuda.amp.GradScaler()
        for images, labels in self.training_loader:
            self.optimizer.zero_grad(set_to_none=True)
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if use_autocast:
                with autocast():
                    outputs = self.model(images)['out']   
                    loss = self.criterion(outputs, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                                
            else:
                outputs = self.model(images)['out']   
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            
            run_loss += loss.item()        
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.numel()
            correct += (predicted == labels).sum().item()
            
                
        epoch_loss = run_loss / len(self.training_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {self.epoch + 1} |   Loss: {epoch_loss}    |   Accuracy: {epoch_acc}%')
        val_loss, val_acc = self.validate(self.val_loader, use_autocast=use_autocast)
        
        # if self.writer is not None: 
        #     self.writer.add_scalars('Loss', {'Training Loss': epoch_loss, 'Validation Loss': val_loss}, self.epoch)
        #     self.writer.add_scalars('Accuracy', {'Training Accuracy': epoch_acc, 'Validation Accuracy': val_acc}, self.epoch)
            
        torch.cuda.empty_cache()
        self.scheduler.step()
        print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
        self.save_model(file_management=False)
        return epoch_loss, epoch_acc, val_loss, val_acc

    def validate(self, val_loader, use_autocast):
        total_loss = 0.0
        correct = 0
        total = 0
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if use_autocast:
                    with autocast():
                        outputs = self.model(images)['out']
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)['out']
                    loss = self.criterion(outputs, labels)
                                
                _, predicted = torch.max(outputs, 1)
                total += labels.numel()
                correct += (predicted == labels).sum().item()                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(val_loader)
            self.val_loss = avg_loss
            val_acc = 100*correct / total 
            print(f'Validation Loss: {avg_loss}')
            return avg_loss, val_acc 
    
    def calculate_miou(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            intersection = 0
            union = 0
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)['out']
                _, predicted = torch.max(outputs, 1)
                intersection += (predicted & labels).float().sum((1, 2))  # Intersection
                union += (predicted | labels).float().sum((1, 2))  # Union
            iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero
            miou = iou.mean().item()
            print(f'Mean IoU: {miou}')
            if hasattr(self, 'writer') and self.writer is not None:
                    self.writer.add_scalars('mIoU', {'Validation': miou}, self.epoch)
            return miou
        
    def calculate_miou2(self, val_loader, num_classes):
        self.model.eval()
        with torch.no_grad():
            intersection = torch.zeros(num_classes).to(self.device)
            union = torch.zeros(num_classes).to(self.device)
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)['out']  # Annahme: Ausgabekanal enthält Klassenlogits
                _, predicted = torch.max(outputs, 1)  # [Batch, H, W]

                for cls in range(num_classes):
                    cls_pred = (predicted == cls)  # Vorhersage für Klasse `cls`
                    cls_label = (labels == cls)    # Ground Truth für Klasse `cls`
                    intersection[cls] += (cls_pred & cls_label).float().sum()
                    union[cls] += (cls_pred | cls_label).float().sum()

            iou = (intersection + 1e-6) / (union + 1e-6)
            miou = iou.mean().item()
            print(f'Mean IoU: {miou}')
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalars('mIoU', {'Validation': miou}, self.epoch)
            return miou

        
        
    def auto_train(self, epochs,  deviation_threshold=0.1, max_deviations=5):
        deviations = 0
        for epoch in range(epochs):
            self.model.train()
            run_loss = 0.0
            self.epoch += 1
            print(f'Epoch {epoch + 1} von {epochs}')
            counter = 0
            correct = 0 
            total = 0 
            for images, labels in tqdm(self.training_loader):
                counter +=1
                self.optimizer.zero_grad(set_to_none=True)
                #print(f'Image {counter} von {len(self.training_loader)}')
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                #print(f'Labels Shape: {labels.shape} and IMage Shape: {images.shape}')
                
                outputs = self.model(images)['out']   
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.numel()
                #_, labels = labels.max(dim=1) WRONG ?!?!?
                #print(f'shape of predicted: {predicted.shape} and shape of labels: {labels.shape}, type of labels: {labels.dtype}')
                correct += (predicted == labels).sum().item()
            
                loss = self.criterion(outputs, labels)            
                loss.backward()
                self.optimizer.step()
                run_loss += loss
                
            epoch_loss = run_loss.item() / len(self.training_loader)
            epoch_acc = 100 * correct / total
            print(f'Epoch {epoch + 1} von {epochs}    |   Loss: {epoch_loss}    |   Accuracy: {epoch_acc}%')
            # Validate the model after each epoch
            val_loss, val_acc = self.validate(self.val_loader, use_autocast=False)
            self.writer.add_scalars('Loss', {'Training Loss': epoch_loss, 'Validation Loss': val_loss}, self.epoch)
            self.writer.add_scalars('Accuracy', {'Training Accuracy': epoch_acc, 'Validation Accuracy': val_acc}, self.epoch)
            
            if self.val_loss > epoch_loss + deviation_threshold:
                print(f'Validation loss deviated too much from training loss in epoch {self.epoch + 1}')
                deviations += 1
                if deviations > max_deviations:
                    print(f'Stopped training due to validation loss deviating too much from training loss in epoch {self.epoch + 1}')
                    break
            torch.cuda.empty_cache()
            self.save_model(file_management=True)
            
    def test(self):
        self.model.eval()
        total = 0
        correct = 0
        test_loss = 0.0 

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)['out'] 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.numel()
                correct += (predicted == labels).sum().item()

                # Calculate and print loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

        test_loss /= len(self.test_loader)
        print('Test Loss: %.3f' % test_loss)

        test_accuracy = 100 * correct / total
        print('Test Accuracy: %.2f %%' % test_accuracy)
        
        self.writer.add_scalar('Test Loss', test_loss, self.epoch)
        self.writer.add_scalar('Test Accuracy', test_accuracy, self.epoch)
        
        
    def inference_tensorboard(self, index):
        self.model.eval()
        mean = torch.tensor([0.2892, 0.3272, 0.2867]).view(1, 3, 1, 1)
        std = torch.tensor([0.1904, 0.1932, 0.1905]).view(1, 3, 1, 1)
        mean = mean.to(self.device)
        std = std.to(self.device)       
        with torch.no_grad():
            images, labels = self.test_loader.dataset[index]  # Get the image and labels at the given index
            images = images.unsqueeze(0).to(self.device)  # Add a batch dimension and move to device
            labels = labels.unsqueeze(0).to(self.device)  # Add a batch dimension and move to device
            outputs = self.model(images)['out']
            _, predicted = torch.max(outputs.data, 1)
            num_classes = outputs.shape[1]
            all_masks = predicted == torch.arange(num_classes, device=self.device)[:, None, None]
            tensor = (images * std) + mean  # Reverse normalization
            tensor = tensor.to(torch.uint8).squeeze(0)
            res_image = draw_segmentation_masks(
                tensor,
                all_masks,
                colors=self.city_label_color_map,
                alpha=0.5
            )

            # Now do the same for the correct labels
            all_masks = labels == torch.arange(num_classes, device=self.device)[:, None, None]
            gt_image = draw_segmentation_masks(
                tensor,
                all_masks,
                colors=self.city_label_color_map,
                alpha=0.5
            )

            # Convert the tensors to PIL Images
            res_image = transforms.ToPILImage()(res_image)
            gt_image = transforms.ToPILImage()(gt_image)

            # Convert the images to 3-channel images for TensorBoard
            res_image_3ch = res_image.convert("RGB")
            gt_image_3ch = gt_image.convert("RGB")

            # Convert the 3-channel images to Tensors
            res_image_tensor = transforms.ToTensor()(res_image_3ch)
            gt_image_tensor = transforms.ToTensor()(gt_image_3ch)

            # Combine the images into a grid
            grid = make_grid([res_image_tensor, gt_image_tensor])

            # Add the image grid to TensorBoard
            self.writer.add_image('Inference Images', grid, self.epoch)

            return None



class MapillaryTrainedModel(TrainedModel):
    def __init__(
        self,
        model_name,
        width,
        height,
        weights_name,
        folder_path="Own_Weights",
        start_epoch="latest",
        pretrained=True,
        writer=None,
        skip_local_load=False
    ):
        # Lade die Mapillary Colormap aus der JSON-Datei
        self.mapillary_label_color_map = self.load_mapillary_colormap("Colormap/mapillary_colormap.json")
        self.num_classes = len(self.mapillary_label_color_map)  # Automatische Anpassung der Klassenanzahl

        self.step = 0
        self.learning_rate = 1e-5
        if writer is not None:
            self.writer = writer

        # Rufe den Eltern-Konstruktor auf und übergebe die neue Klassenanzahl
        super().__init__(
            model_name,
            width=width,
            height=height,
            pretrained=pretrained,
            num_classes=self.num_classes, 
            mapillary=True,
        )

        # Optional: Preprocessing-Pipeline sicherstellen
        self.preprocess = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.PILToTensor(),
            SemanticSegmentation(resize_size=520),
        ])

        # Restlicher Initialisierungscode bleibt unverändert

    @staticmethod
    def load_mapillary_colormap(colormap_path):
        """ Lädt die Farbzuordnung aus der JSON-Datei. """
        with open(colormap_path, "r") as file:
            colormap = json.load(file)

        # Konvertiere die Farbwerte in ein Tupel (R, G, B)
        return [tuple(color) for color in colormap.values()]

    def own_model_inference_live_no_grad(self, image):
        """ Führt eine segmentierte Vorhersage aus und visualisiert das Ergebnis. """
        self.model.eval()
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
                colors=self.mapillary_label_color_map,  # Verwende die neue Mapillary-Farbkarte
                alpha=0.9
            )
            res_image = self.tensor_to_image(res_image)
            return res_image




class CustomDataSet(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.mean = (0.2892, 0.3272, 0.2867)
        self.std = (0.1904, 0.1932, 0.1905)
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        
        self.transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=90, p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, p=0.5),
            RandomCrop(height=600, width=600, p=0.5),
            Resize(520, 520),  # Resize the image and mask
            Normalize(mean=self.mean, std=self.std),  # Normalize the image
            ToTensorV2()  # Convert the image and mask to PyTorch tensors
        ])
        
        self.image_files = os.listdir(image_dir)
        self.annotation_files = os.listdir(annotation_dir)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):        
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        annotation_name = os.path.join(self.annotation_dir, self.annotation_files[idx])

        image = Image.open(img_name).convert("RGB")            
        annotation = Image.open(annotation_name).convert("L")     

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(annotation))
            image = augmented['image']
            annotation = augmented['mask']
            
        annotation = annotation.long()
        
        return image, annotation
                
    #    # Convert annotation to tensor and add an extra dimension
    #     annotation = torch.from_numpy(np.array(annotation)).long().unsqueeze(0)

    #     # Create a tensor for one-hot encoding
    #     one_hot_annotation = torch.zeros(20, *annotation.shape[1:])

    #     # Perform one-hot encoding
    #     one_hot_annotation.scatter_(0, annotation.long(), 1)

    #     return image, one_hot_annotation

class K_Fold_Dataset:
    def __init__(self, image_dir, annotation_dir, k_fold_csv_dir, leave_out_fold, num_classes=20):
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
        self.test_dataset = self.TestDataset(self.test_files, image_dir, annotation_dir)
        
    def check_for_data_leaks(self):
        train_files_set = set(file for sublist in self.train_files for file in sublist)
        val_files_set = set(file for sublist in self.val_files for file in sublist)
        test_files_set = set(file for sublist in self.test_files for file in sublist)

        # Check for overlaps between sets
        train_val_overlap = train_files_set & val_files_set
        train_test_overlap = train_files_set & test_files_set
        val_test_overlap = val_files_set & test_files_set

        if train_val_overlap:
            print(f"Data leak between training and validation sets: {train_val_overlap}")
            sys.exit()
        if train_test_overlap:
            print(f"Data leak between training and test sets: {train_test_overlap}")
            sys.exit()
        if val_test_overlap:
            print(f"Data leak between validation and test sets: {val_test_overlap}")
            sys.exit()

        if not train_val_overlap and not train_test_overlap and not val_test_overlap:
            print("No data leaks found.")
        
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
    
    class TestDataset(CustomDataSet):
        def __init__(self, test_files, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            self.image_files = [file[0] for file in test_files]
            self.annotation_files = [file[0] for file in test_files]
                        
            self.transform = Compose([
            Resize(520, 520),  # Resize the image and mask
            Normalize(mean=self.mean, std=self.std),  # Normalize the image
            ToTensorV2()  # Convert the image and mask to PyTorch tensors
        ])
            
        def __getitem__(self, idx):        
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            annotation_name = os.path.join(self.annotation_dir, self.annotation_files[idx])

            image = Image.open(img_name).convert("RGB")            
            annotation = Image.open(annotation_name).convert("L")     

            if self.transform:
                augmented = self.transform(image=np.array(image), mask=np.array(annotation))
                image = augmented['image']
                annotation = augmented['mask']
                
            annotation = annotation.long()
            
            return image, annotation
        
class MapillaryDataLoader:
    def __init__(self, train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir, val_split=0.2):
        """
        Args:
            train_images_dir: Pfad zu den Trainingsbildern.
            train_annotations_dir: Pfad zu den Trainingsannotations (bereits konvertiert).
            val_images_dir: Pfad zu den Testbildern (ehemals Val-Bilder).
            val_annotations_dir: Pfad zu den Testannotations.
            val_split: Anteil des Trainingssets, der als Validierungssplit genutzt wird (Standard: 20%).
        """
        self.train_images_dir = train_images_dir
        self.train_annotations_dir = train_annotations_dir
        # Das separate Val-Verzeichnis wird als Testset genutzt, da für das Testset keine gelabelten Daten erhältnich waren
        self.test_images_dir = val_images_dir         
        self.test_annotations_dir = val_annotations_dir

        # Lade Dateinamen aus dem Trainingsordner
        train_img_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith(('.png', '.jpg'))])
        train_ann_files = sorted([f for f in os.listdir(train_annotations_dir) if f.endswith('.png')])
        
        # Erstelle ein DataFrame mit den Trainingsdaten und führe einen 80/20 Split durch
        df = pd.DataFrame({'image': train_img_files, 'annotation': train_ann_files})
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
        self.train_files = train_df.values.tolist()  # Jede Zeile: [image_filename, annotation_filename]
        self.val_files = val_df.values.tolist()
        
        # Für das Testset werden die Dateien aus dem separaten Testordner geladen
        test_img_files = sorted([f for f in os.listdir(self.test_images_dir) if f.endswith(('.png', '.jpg'))])
        test_ann_files = sorted([f for f in os.listdir(self.test_annotations_dir) if f.endswith('.png')])
        self.test_files = list(zip(test_img_files, test_ann_files))
        
        # Erstelle die Dataset-Instanzen
        self.train_dataset = self.TrainDataset(self.train_files, train_images_dir, train_annotations_dir)
        self.val_dataset = self.ValDataset(self.val_files, train_images_dir, train_annotations_dir)
        self.test_dataset = self.TestDataset(self.test_files, self.test_images_dir, self.test_annotations_dir)
        
    class TrainDataset(CustomDataSet):
        def __init__(self, file_list, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            # Überschreibe die Dateilisten basierend auf dem 80/20-Split
            self.image_files = [row[0] for row in file_list]
            self.annotation_files = [row[1] for row in file_list]
            
    class ValDataset(CustomDataSet):
        def __init__(self, file_list, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            self.image_files = [row[0] for row in file_list]
            self.annotation_files = [row[1] for row in file_list]
            
    class TestDataset(CustomDataSet):
        def __init__(self, file_list, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            self.image_files = [row[0] for row in file_list]
            self.annotation_files = [row[1] for row in file_list]
            # Für das Testset sollen nur minimale Transformationen (Resize, Normalization, ToTensor) verwendet werden.
            from albumentations import Compose, Resize, Normalize
            from albumentations.pytorch import ToTensorV2
            self.transform = Compose([
                Resize(520, 520),
                Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])