from functools import partial
import os
import torch
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from Helper.ml_models import * 
from Helper.Helper_functions import *
from ray.tune.search.optuna import OptunaSearch
import json
from datetime import datetime


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
    
    
all_models = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']
not_yet_studied = ['fcn_resnet50', 'fcn_resnet101']
test_epochs = 60

k_fold_dataset = K_Fold_Dataset('/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/images',
                         '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/semantic',
                         k_fold_csv_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Daten/CityscapesDaten',
                         leave_out_fold=0,
                         )

k_fold_dataset.check_for_data_leaks()               
        
        
model = all_models[0]


def train_hyper(config):
    try:
        folder_path = '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Hyperparameter'
        current_time = datetime.now().strftime('%H_%M_%S')
        create_raytune_model_directory(path = f'{folder_path}', model=f'{model}_{current_time}')
        hyper_model = TrainedModel(model, 2048, 1024, weights_name=f'', folder_path=f'{folder_path}/{model}_{current_time}', start_epoch='latest')
        hyper_model.prepare_model_training(dataset_train=k_fold_dataset.train_dataset,
                                                dataset_val=k_fold_dataset.val_dataset,
                                                dataset_test=k_fold_dataset.test_dataset,
                                                batch_size=config['batch_size'], 
                                                shuffle=True, 
                                                learning_rate=config['learning_rate'],
                                                weight_decay=config['weight_decay'], 
                                                num_workers=4, 
                                                pin_memory=True,
                                                ray_tune=True,
                                                )

        epoch_loss, epoch_acc = hyper_model.train() 
        miou = hyper_model.calculate_miou_miou(k_fold_dataset.val_dataset)
        train.report({"acc":epoch_acc, "miou":miou})
    except RuntimeError as e:
        if "out of memory" in str(e):
            train.report({"acc":0, "miou":0})
        else:
            raise e  
        
        

config = {
    "learning_rate": tune.loguniform(1e-8, 1e-2),
    'batch_size': tune.choice([2,4,6,8,10]),
    "weight_decay": tune.loguniform(1e-6, 1e-1)
}

# Define the scheduler and reporter
scheduler = ASHAScheduler(
    metric="acc",
    mode="max",
    max_t=80,
    grace_period=5,
    reduction_factor=2
)


optuna_search = OptunaSearch(
    metric="acc",
    mode="max"
)


reporter = CLIReporter(metric_columns=["loss", "acc", "training_iteration"])
analysis = tune.run(train_hyper,
                    config=config,
                    resources_per_trial={"cpu": 6, "gpu": 1},
                    scheduler=scheduler,    
                    progress_reporter=reporter, 
                    resume=True,
                    local_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/HyperparameterLOG', 
                    search_alg=optuna_search,
                    num_samples=50,
                    )

best_config = analysis.get_best_config(metric="acc", mode="max")

print("Best hyperparameters found were: ", best_config)
# Save the best configuration to a JSON file
with open('best_config.json', 'w') as json_file:
    json.dump(best_config, json_file)

print("Best configuration saved to best_config.json.")