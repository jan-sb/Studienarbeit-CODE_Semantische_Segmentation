import ray
from ray import tune

ray.shutdown()
ray.init()

# Pfad muss der gleiche sein, den Sie auch in `storage_path` angegeben haben
storage_path = "/home/jan/studienarbeit/Studienarbeit-CODE_Semantische_Segmentation/HyperparameterLOG/deeplabv3_resnet50"

# Restore Tuner
tuner = tune.Tuner.restore(
    path=storage_path,
    resume_unfinished=True,   # unfinished Trials wieder aufnehmen
    resume_errored=False,     # Trials, die per OOM gecrashed sind, ignorieren
)

analysis = tuner.fit()

best_result = analysis.get_best_result(metric="val_loss", mode="min")
print("Best config: ", best_result.config)
