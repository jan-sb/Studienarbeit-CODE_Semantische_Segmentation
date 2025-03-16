from Helper.ml_models import MapillaryDataLoader, MapillaryTrainedModel

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

best_config_path = "FINAL_DATEN/best_configs_Map_BIG.json"


# Mapillary-Daten laden 
# UNBEDINGT (!!!) volle Pfade nutzen, da Raytune aus dem Basisverzeichnis startet
# und sonst die Pfade nicht findet
mapillary_loader = MapillaryDataLoader(
    train_images_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Mapillary_Vistas/training/images',
    train_annotations_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Mapillary_Vistas/training/v2.0/labels_small',
    val_images_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Mapillary_Vistas/validation/images',
    val_annotations_dir='/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/Mapillary_Vistas/validation/v2.0/labels_small'
)

def make_directory(model):
    dir_name = f'Hyperparameter/{model}'
    os.makedirs(dir_name, exist_ok=True)
    
    
    
modells_to_study = [
    'deeplabv3_resnet50', 
    'deeplabv3_resnet101', 
    'fcn_resnet50',
    'fcn_resnet101'
]

def train_hyper(config, checkpoint_dir=None):  
    try:
        make_directory(model)

        # Nutze jetzt die MapillaryTrainedModel-Klasse!
        hyper_model = MapillaryTrainedModel(
            model_name=model,
            width=2048,
            height=1024,
            weights_name='',
            folder_path=f'Hyperparameter/{model}',
            start_epoch='latest'
        )
        
        # Number of classes check
        print(f"[INIT] Modell '{model}' initialisiert mit {hyper_model.num_classes} Klassen.")


        # Falls ein Checkpoint existiert, lade ihn
        if checkpoint_dir:
            with get_checkpoint().as_directory() as checkpoint_dir_path:
                with open(os.path.join(checkpoint_dir_path, 'checkpoint.pkl'), 'rb') as fp:
                    checkpoint = pickle.load(fp)
                    hyper_model.model.load_state_dict(checkpoint["model_state"])
                    hyper_model.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 0

        # Nutze jetzt den Mapillary-Dataloader
        hyper_model.prepare_model_training(
            dataset_train=mapillary_loader.train_dataset,
            dataset_val=mapillary_loader.val_dataset,
            batch_size=int(config['batch_size']),
            val_batch_size=int(config['batch_size']),
            shuffle=True,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            num_workers=4,
            pin_memory=True,
            ray_tune=True,
        )

        max_epochs = min(config.get("max_epochs", 100), 100)

        for epoch in range(start_epoch, max_epochs):
            epoch_loss, epoch_acc, val_loss, val_acc = hyper_model.train(use_autocast=config['auto_cast'])

            print(
                f"Epoch: {epoch}, "
                f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

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
                        "val_acc": val_acc,
                        "training_iteration": epoch,
                        "num_classes": hyper_model.num_classes,
                    },
                    checkpoint=checkpoint_obj
                )

        print("Training completed successfully!")

    except RuntimeError as e:
        if "out of memory" in str(e):
            train.report({"loss": float('inf'), "val_loss": float('inf'), "train_acc": 0.0, "val_acc": 0.0})
        else:
            raise e



for model in modells_to_study:
    ray.shutdown()
    ray.init() 

    # Define your parameter search space
    config = {
        "learning_rate": tune.loguniform(0.0001, 0.00011),
        "batch_size": 6,
        "weight_decay": 0,
        "auto_cast": True,
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
            num_samples=1,
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
            name=f"{model}",
            storage_path="/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/HyperparameterLOG",
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            progress_reporter=CLIReporter(
                metric_columns=["loss", "val_loss", "train_acc", "val_acc", "training_iteration", "num_classes"],
            ),
        ),
    )

    analysis = tuner.fit()


    # Fetch the best trial
    best_result = analysis.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    print("Best trial config: ", best_config)
    
    
    # Speichern der besten Konfiguration in die JSON-Datei
    try:
        # Falls die Datei existiert, lade den aktuellen Inhalt
        if os.path.exists(best_config_path):
            with open(best_config_path, "r") as file:
                all_configs = json.load(file)
        else:
            all_configs = {}

        # Speichere die neue beste Konfiguration unter dem Modellnamen
        all_configs[model] = best_config

        # Schreibe die aktualisierten Konfigurationen zurück in die Datei
        with open(best_config_path, "w") as file:
            json.dump(all_configs, file, indent=4)

        print(f"Beste Konfiguration für {model} gespeichert.")

    except Exception as e:
        print(f"Fehler beim Speichern der besten Konfiguration für {model}: {e}")
