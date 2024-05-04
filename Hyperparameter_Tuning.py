from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.tune import CLIReporter
from Helper.ml_models import * 
import json


def load_data(image_dir='CityscapesDaten/images', annotation_dir='CityscapesDaten/semantic'):
    trainset = CustomDataSet(image_dir=image_dir, annotation_dir=annotation_dir)

    # If you have a separate set of images and annotations for testing, you can create a testset in a similar way:
    # testset = CustomDataSet(image_dir=test_image_dir, annotation_dir=test_annotation_dir)

    # If you don't have a separate test set, you can split the trainset into a training set and a test set:
    train_size = int(0.8 * len(trainset))
    test_size = len(trainset) - train_size
    trainset, testset = torch.utils.data.random_split(trainset, [train_size, test_size])

    return trainset, testset

def make_directory(model):
    dir_name = f'Hyperparameter/{model}'
    os.makedirs(dir_name, exist_ok=True)
    

all_models = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'fcn_resnet50', 'fcn_resnet101', 'lraspp_mobilenet_v3_large']
test_epochs = 20

k_fold_dataset = K_Fold_Dataset(
                        image_dir='CityscapesDaten/images',
                        annotation_dir='CityscapesDaten/semantic',
                        k_fold_csv_dir='Daten/CityscapesDaten',
                        leave_out_fold=0,
                        )
k_fold_dataset.check_for_data_leaks()        

model = all_models[0]


def train_hyper(config):
    
    make_directory(model)
    img_path = '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/images' 
    anno_path = '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/semantic'
    csv_path = '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Daten/CityscapesDaten'
    print(f'IMGPATH:{os.access(img_path, os.R_OK)}, ANNOPATH:{os.access(anno_path, os.R_OK)}, CSVPATH:{os.access(csv_path, os.R_OK)}')
    print(f'(\n \n {os.getcwd()} \n \n)')
    
    
    k_fold_dataset = K_Fold_Dataset(
                        image_dir=img_path,
                        annotation_dir=anno_path,
                        k_fold_csv_dir=csv_path,
                        leave_out_fold=0,
                        )
    hyper_model = TrainedModel(model, 2048, 1024, weights_name='', folder_path=f'Hyperparameter/{model}', start_epoch='latest')
    hyper_model.prepare_model_training(dataset_train=k_fold_dataset.train_dataset,
                                                dataset_val=k_fold_dataset.val_dataset,
                                                batch_size=int(config['batch_size']), 
                                                shuffle=True, 
                                                learning_rate=config['lr'], 
                                                momentum=config['momentum'],
                                                weight_decay=config['weight_decay'],)

    
    hyper_model.train(1)  # Train for one epoch
    val_loss = hyper_model.validate(k_fold_dataset.val_dataset)  # Validate on the validation dataset
    miou = hyper_model.calculate_miou_miou(k_fold_dataset.val_dataset)
    tune.report(loss=val_loss, miou=miou)
        

config = {
    "lr": tune.loguniform(1e-12, 1e-2),
    "batch_size": tune.choice([2, 4, 6]),
    "momentum": tune.uniform(0.1, 0.9),
    "weight_decay": tune.loguniform(1e-6, 1e-1)
}

scheduler = ASHAScheduler(
    metric="miou",
    mode="max",
    max_t=30,
    grace_period=3,
    reduction_factor=2
)

reporter = CLIReporter(metric_columns=["loss", "miou", "training_iteration"])

# config = {  
#         'batch_size': 2,
#         'lr' : 0.001,
#         'momentum' : 0.9,
#         'weight_decay' : 0.0005,    
# }

# train_hyper(config)

# sys.exit()
# file_path = 'CityscapesDaten/images/0000088_01.png' 
# print(f'FILEPATH VALID {os.access(file_path, os.R_OK)}')

analysis = tune.run(
                    train_hyper,
                    config=config,
                    resources_per_trial={"cpu": 1, "gpu": 1},
                    scheduler=scheduler,
                    progress_reporter=reporter)

# Get the best trial
best_trial = analysis.get_best_trial("loss")

# Get the best trial's parameters
best_params = best_trial.config

# Save the parameters to a JSON file
with open('/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Daten/best_params.json', 'w') as f:
    json.dump(best_params, f)