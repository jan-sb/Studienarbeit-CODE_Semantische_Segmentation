from functools import partial
from tqdm import tqdm
import os
import tempfile
import torch

from Helper.ml_models import * 
import json
from datetime import datetime
import ray.cloudpickle as pickle

from ray import tune, train
from ray.air.config import CheckpointConfig
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray

from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler

import ray
ray.shutdown()
ray.init() 

def make_directory(model):
    dir_name = f'Hyperparameter/{model}'
    os.makedirs(dir_name, exist_ok=True)

# Variables
all_models = [
    'deeplabv3_resnet50', 
    'deeplabv3_resnet101', 
    'deeplabv3_mobilenet_v3_large', 
    'lraspp_mobilenet_v3_large'
]
not_yet_studied = ['fcn_resnet50', 'fcn_resnet101']

k_fold_dataset = K_Fold_Dataset(
    '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/images',
    '/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/CityscapesDaten/semantic',
    k_fold_csv_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Daten/CityscapesDaten',
    leave_out_fold=0,
)
k_fold_dataset.check_for_data_leaks()         

model = all_models[1]      

def train_hyper(config, checkpoint_dir=None):  
    try:
        make_directory(model)
        hyper_model = TrainedModel(
            model,
            2048,
            1024,
            weights_name='',
            folder_path=f'Hyperparameter/{model}',
            start_epoch='latest'
        )
        
        # Load checkpoint if available
        if checkpoint_dir:
            with get_checkpoint().as_directory() as checkpoint_dir_path:
                with open(os.path.join(checkpoint_dir_path, 'checkpoint.pkl'), 'rb') as fp:
                    checkpoint = pickle.load(fp)
                    hyper_model.model.load_state_dict(checkpoint["model_state"])
                    hyper_model.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 0

        # Prepare data/dataloaders
        hyper_model.prepare_model_training(
            dataset_train=k_fold_dataset.train_dataset,
            dataset_val=k_fold_dataset.val_dataset,
            dataset_test=k_fold_dataset.test_dataset,
            batch_size=int(config['batch_size']),
            val_batch_size=int(config['batch_size']),
            shuffle=True,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            num_workers=4,
            pin_memory=True,
            ray_tune=True,
        )

        # Use a dynamic number of epochs up to 100
        max_epochs = min(config.get("max_epochs", 100), 100)

        for epoch in range(start_epoch, max_epochs):
            # Train and evaluate
            epoch_loss, epoch_acc, val_loss, val_acc = hyper_model.train(use_autocast=config['auto_cast'])
            
            print(
                f"Epoch: {epoch}, "
                f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save checkpoint data for resumption
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_data = {
                    "model_state": hyper_model.model.state_dict(),
                    "optimizer_state": hyper_model.optimizer.state_dict(),
                    "epoch": epoch,
                }
                with open(os.path.join(tmp_dir, 'checkpoint.pkl'), 'wb') as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint_obj = Checkpoint.from_directory(tmp_dir)
                train.report(
                    {
                        "loss": epoch_loss,
                        "val_loss": val_loss,
                        "train_acc": epoch_acc,
                        "val_acc": val_acc
                    },
                    checkpoint=checkpoint_obj
                )

        # Final checkpoint
        final_checkpoint_dir = tempfile.mkdtemp()
        final_checkpoint_data = {
            "model_state": hyper_model.model.state_dict(),
            "optimizer_state": hyper_model.optimizer.state_dict(),
            "epoch": max_epochs,
        }
        with open(os.path.join(final_checkpoint_dir, 'final_checkpoint.pkl'), 'wb') as fp:
            pickle.dump(final_checkpoint_data, fp)

        final_checkpoint = Checkpoint.from_directory(final_checkpoint_dir)
        train.report(
            {
                "loss": epoch_loss,
                "val_loss": val_loss,
                "train_acc": epoch_acc,
                "val_acc": val_acc
            },
            checkpoint=final_checkpoint
        )

    except RuntimeError as e:
        # Handle out-of-memory errors
        if "out of memory" in str(e):
            train.report({"loss": float('inf'), "val_loss": float('inf'), "train_acc": 0.0, "val_acc": 0.0})
        else:
            raise e


# Define your parameter search space
config = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([4, 8, 16]),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "auto_cast": tune.choice([True, False]),
    "max_epochs": 100,
}

# Create an OptunaSearch object (which implements TPE)
search_alg = OptunaSearch(
    sampler=TPESampler(seed=42),
    metric="val_loss",
    mode="min",
)

# Wrap your training function to request 1 CPU and 1 GPU
train_hyper_with_resources = tune.with_resources(
    train_hyper,
    resources={"cpu": 1, "gpu": 1}
)

tuner = tune.Tuner(
    train_hyper_with_resources,
    param_space=config,
    tune_config=tune.TuneConfig(
        num_samples=50,
        search_alg=search_alg,
        scheduler=ASHAScheduler(
            max_t=100,
            grace_period=5,
            reduction_factor=3,
        ),
        metric="val_loss",
        mode="min",
    ),
    run_config=train.RunConfig(
        name="Hyperparameter_Tuning_Deeplabv3",
        # Wichtig: Speichert alles im storage_path
        storage_path="/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/HyperparameterLOG/deeplabv3_resnet50",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
        progress_reporter=CLIReporter(
            metric_columns=["loss", "val_loss", "train_acc", "val_acc", "training_iteration"]
        ),
    ),
)

analysis = tuner.fit()


# Fetch the best trial
best_result = analysis.get_best_result(metric="val_loss", mode="min")
best_config = best_result.config
print("Best trial config: ", best_config)

