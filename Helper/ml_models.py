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

    def __init__(self, model_name, width, height, weights_name, folder_path = 'Own_Weights', start_epoch='latest', pretrained=True, writer = None):
        self.city_label_color_map = [
            (128, 64, 128), #ID__0, road
            (244, 35, 232), #ID__1, sidewalk
            (70, 70, 70), #ID__2, building
            (102, 102, 156), #ID__3, wall
            (190, 153, 153), #ID__4, fence
            (153, 153, 153), #ID__5, pole
            (250, 170, 30), #ID__6, traffic light
            (220, 220, 0), #ID__7, traffic sign
            (107, 142, 35), #ID__8, vegetation
            (152, 251, 152), #ID__9, terrain
            (70, 130, 180), #ID__10, sky
            (220, 20, 60), #ID__11, person
            (255, 0, 0), #ID__12, rider
            (0, 0, 142), #ID__13, car
            (0, 0, 70), #ID__14, truck
            (0, 60, 100), #ID__15, bus
            (0, 80, 100), #ID__16, train
            (0, 0, 230), #ID__17, motorcycle
            (119, 11, 32), #ID__18, bicycle
            (0, 0, 0), #ID__19, unlabeled   #gets relabeled in annotation call, call and groundtruth need adjustments
        ]
        self.num_classes = len(self.city_label_color_map)
        self.step = 0
        self.learning_rate = 1*10**(-5)
        if writer is not None: 
            self.writer = writer
        
        
        
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
        self.val_loss = 0
        self.old_val_loss = self.val_loss

        # Make Folder for the Trained Weights
        self.folder_path = folder_path
        self.weights_name = weights_name
        self.model_folder_path = os.path.join(self.folder_path, self.weights_name)


        # Loading the Model
        path_to_latest = self.model_folder_path+ f'/{self.weights_name}_latest_{self.model_name}.pth'
        path_to_epoch = self.model_folder_path+ f'/{self.weights_name}_epoch-{self.epoch}_{self.model_name}.pth'
        if os.path.exists(path_to_latest) and start_epoch == 'latest':
            try:
                checkpoint = torch.load(path_to_latest)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.val_loss = checkpoint['val_loss']
                self.step = checkpoint['step']
            except:
                print(f'Error loading Model with Epoch latest in Class TrainedModel')
                sys.exit()
        elif os.path.exists(path_to_epoch) and start_epoch != 'latest':
            try:
                checkpoint = torch.load(path_to_epoch)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.val_loss = checkpoint['val_loss']
                self.step = checkpoint['step']
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
                self.val_loss = 0
                self.save_model(file_management=False)
                #latest_file_path = self.model_folder_path + f'_latest_{self.model_name}.pth'
                checkpoint = torch.load(path_to_latest)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.val_loss = checkpoint['val_loss']
                self.step = checkpoint['step']
                print(f'Successfully loaded Model')
            except:
                print(f'Failed to initialise new model')
                sys.exit()
                
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'{folder_path}/{weights_name}/runs/{current_time}'

        self.old_val_loss = self.val_loss
        self.model.eval()
        
    def inference(self, image):
        self.model.eval()
        with torch.no_grad():
            #tensor = self.image_preprocess(image)
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



    def prepare_model_training(self, dataset_train=None, dataset_val= None, batch_size=3, shuffle=True, learning_rate= 1*10**(-5), momentum=0.9, weight_decay=0.001, num_workers = 0, pin_memory = False):
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
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory,
                                        )
            print(f'Validation Dataset prepared')

        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #self.optimizer = torch.optim.Adadelta(self.model.parameters())

    def save_model(self, file_management=False):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

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

    def train(self, epochs):
        self.model.train()
        torch.cuda.empty_cache()
        for epoch in range(epochs):
            run_loss = 0.0
            self.epoch += 1
            for images, labels in self.training_loader:
                self.optimizer.zero_grad(set_to_none=True)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)['out']                    
                _, labels = labels.max(dim=1)
                
                loss = self.criterion(outputs, labels)            
                loss.backward()
                
                self.optimizer.step()
                run_loss += loss.item()
                
                # For Tensorboard
                
                self.step = epoch * len(self.training_loader) + 1
                self.writer.add_scalars('Loss', {'Train': loss.item()}, self.step)
                
            epoch_loss = run_loss / len(self.training_loader)
            print(f'Epoch {epoch + 1} von {epochs}    |   Loss: {epoch_loss}')
            self.writer.add_scalar('Epoch Train Loss', epoch_loss, self.epoch)
        self.save_model(file_management=True)
        
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
            self.val_loss = avg_loss
            print(f'Validation Loss: {avg_loss}')
            self.writer.add_scalars('Loss', {'Validation': avg_loss}, self.epoch)
            return avg_loss
    
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
            self.writer.add_scalars('mIoU', {'Validation': miou}, self.epoch)
            return miou
        
        
    def auto_train(self, epochs, l2_lambda=0.001, deviation_threshold=0.1, max_deviations=5):
        deviations = 0
        for epoch in range(epochs):
            self.model.train()
            run_loss = 0.0
            self.epoch += 1
            print(f'Epoch {epoch + 1} von {epochs}')
            counter = 0
            for images, labels in tqdm(self.training_loader):
                counter +=1
                self.optimizer.zero_grad(set_to_none=True)
                #print(f'Image {counter} von {len(self.training_loader)}')
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                #print(f'Labels Shape: {labels.shape} and IMage Shape: {images.shape}')
                
                outputs = self.model(images)['out']                    
                _, labels = labels.max(dim=1)
                
                loss = self.criterion(outputs, labels)            
                loss.backward()
                self.optimizer.step()
                run_loss += loss
                
                #print(f'Loss: {loss.item()}')                
                # For Tensorboard                
                #step = epoch * len(self.training_loader) + counter
                #self.writer.add_scalar('Training Loss', loss.item(), step)
                
            epoch_loss = run_loss.item() / len(self.training_loader)
            print(f'Epoch {epoch + 1} von {epochs}    |   Loss: {epoch_loss}')
            # Validate the model after each epoch
            self.val_loss = self.validate(self.val_loader)
            self.writer.add_scalars('Epoch and Validation Loss', {'Epoch Loss': epoch_loss, 'Validation Loss': self.val_loss}, self.epoch)
            
            if self.val_loss > epoch_loss + deviation_threshold:
                print(f'Validation loss deviated too much from training loss in epoch {self.epoch + 1}')
                deviations += 1
                if deviations > max_deviations:
                    print(f'Stopped training due to validation loss deviating too much from training loss in epoch {self.epoch + 1}')
                    break
            torch.cuda.empty_cache()
            self.save_model()
            
        #self.writer.flush()
        #self.writer.close()
        
        
    def inference_tensorboard(self, image): # does not work as intended yet
        self.model.eval()
        with torch.no_grad():
            tensor = self.image_preprocess(image)
            output = self.model(tensor)['out'].to(self.device)
            num_classes = output.shape[1]
            all_masks = output.argmax(1) == torch.arange(num_classes, device=self.device)[:, None, None]
            tensor = tensor.to(torch.uint8).squeeze(0)
            res_image = draw_segmentation_masks(
                tensor,
                all_masks,
                colors=self.city_label_color_map,
                alpha=0.9
            )

            # Convert the tensor to a PIL Image
            res_image = transforms.ToPILImage()(res_image)

            # Convert the image to a 3-channel image for TensorBoard
            res_image_3ch = res_image.convert("RGB")

            # Convert the 3-channel image to a Tensor
            res_image_tensor = transforms.ToTensor()(res_image_3ch)

            # Add the image to TensorBoard
            self.writer.add_image('Inference Images', res_image_tensor, self.epoch)

            return None


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
                
       # Convert annotation to tensor and add an extra dimension
        annotation = torch.from_numpy(np.array(annotation)).long().unsqueeze(0)

        # Create a tensor for one-hot encoding
        one_hot_annotation = torch.zeros(20, *annotation.shape[1:])

        # Perform one-hot encoding
        one_hot_annotation.scatter_(0, annotation.long(), 1)

        return image, one_hot_annotation

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

            # self.transform = transforms.Compose([
            #     transforms.RandomResizedCrop(224),  # Random Cropping
            #     transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random Shifting
            #     transforms.ToTensor(), # converte to Tensor
            #     #transforms.Normalize(mean=self.mean, std=self.std),
            # ])

        # def __getitem__(self, index):
        #     img_name = os.path.join(self.image_dir, self.image_files[index])
        #     annotation_name = os.path.join(self.annotation_dir, self.annotation_files[index])

        #     image = Image.open(img_name).convert("RGB")
        #     annotation = Image.open(annotation_name).convert("L")

        #     # Apply transformations
        #     image = self.transform(image)
        #     annotation = self.transform(annotation)

        #     return image, annotation
        
            
    class ValDataset(CustomDataSet):
        def __init__(self, val_files, image_dir, annotation_dir):
            super().__init__(image_dir, annotation_dir)
            self.image_files = [file[0] for file in val_files]
            self.annotation_files = [file[0] for file in val_files]