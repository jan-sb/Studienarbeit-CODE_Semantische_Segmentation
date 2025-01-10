from functools import partial
import os
import torch
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from Helper.ml_models import * 
import json
from datetime import datetime

from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler



def make_directory(model):
    dir_name = f'Hyperparameter/{model}'
    os.makedirs(dir_name, exist_ok=True)
    

# Variables
all_models = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']
not_yet_studied = ['fcn_resnet50', 'fcn_resnet101']

k_fold_dataset = K_Fold_Dataset('/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/images',
                         '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/semantic',
                         k_fold_csv_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Daten/CityscapesDaten',
                         leave_out_fold=0,
                         )

k_fold_dataset.check_for_data_leaks()               

model = all_models[0]

def train_hyper(config, checkpoint_dir=None):  
    try:
        make_directory(model)
        hyper_model = TrainedModel(model, 2048, 1024, weights_name='', folder_path=f'Hyperparameter/{model}', start_epoch='latest')
        
        # Checkpoint laden, falls vorhanden
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pth")):
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
            hyper_model.model.load_state_dict(checkpoint["model_state"])
            hyper_model.optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0
        
        
        hyper_model.prepare_model_training(
            dataset_train=k_fold_dataset.train_dataset,
            dataset_val=k_fold_dataset.val_dataset,
            dataset_test=k_fold_dataset.test_dataset,
            batch_size=int(config['batch_size']), 
            shuffle=True, 
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'], 
            num_workers=4, 
            pin_memory=True,
            ray_tune=True,
            )

        
        EPOCHS = 20 

        for epoch in range(start_epoch, EPOCHS):
            epoch_loss, epoch_acc = hyper_model.train(use_autocast=config['auto_cast']) 
            miou = hyper_model.calculate_miou(k_fold_dataset.val_dataset)
            with tune.checkpoint_dir(epoch) as cp_dir:
                hyper_model.save_model(file_management=False, save_path=cp_dir)
            tune.report(loss=epoch_loss, miou=miou, acc=epoch_acc)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            tune.report(loss=float('inf'), miou=0)  
        else:
            raise e  
        
config = {
    "learning_rate": tune.loguniform(1e-12, 1e-2),
    'batch_size': tune.choice([2,4,6,8,12,14,16]),
    "weight_decay": tune.loguniform(1e-6, 1e-1), 
    "auto_cast": tune.choice([True, False]),
}

analysis = tune.run(
    train_hyper,
    config=config,
    resources_per_trial={"cpu": 6, "gpu": 1},
    scheduler=ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=20,
        grace_period=5,
        reduction_factor=3,
    ),
    progress_reporter=CLIReporter(metric_columns=["loss", "miou", "acc", "training_iteration"]),
    local_dir=f"/home/jan/studienarbeit/HyperparameterLOG/{model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    search_alg=OptunaSearch(
        metric="loss",
        mode="min",
        sampler=TPESampler(seed=42),
    ),
    num_samples=100,
    #checkpoint_config=train.CheckpointConfig(
        #checkpoint_frequency=5,
        #checkpoint_at_end=True,
    #),
    resume=True,
)

print("Best hyperparameters found were: ", analysis.best_config)

best_config = analysis.best_config

# Save the best configuration to a JSON file
with open('hyper_best_config.json', 'w') as json_file:
    json.dump(best_config, json_file)

print("Best configuration saved to best_config.json.")

# Speichere alle getesteten Konfigurationen und Ergebnisse
all_trials = analysis.trials
with open('hyper_all_trials.json', 'w') as json_file:
    json.dump([trial.config for trial in all_trials], json_file)

print("All configurations saved to all_trials.json.")
