from Helper.ml_models import *
from Helper.Helper_functions import * 
from torchvision.models.segmentation import *

deeplv3 = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']
fcn = ['fcn_resnet50', 'fcn_resnet101']
lrsapp = ['lraspp_mobilenet_v3_large']



for model in deeplv3:
    for i in range(5): 
        trained_model = TrainedModel(model, 2048, 1024, f'{model}_k_fold_2_{i}', start_epoch='latest')
        k_fold_dataset = K_Fold_Dataset(image_dir='CityscapesDaten/images',
                                        annotation_dir='CityscapesDaten/semantic',
                                        k_fold_csv_dir='Daten/CityscapesDaten',
                                        leave_out_fold=i)
        
        trained_model.prepare_model_training(dataset_train=k_fold_dataset.train_dataset,
                                             dataset_val=k_fold_dataset.val_dataset,
                                             batch_size=2, 
                                             shuffle=True, 
                                             learning_rate=1*10**(-5), 
                                             momentum=0.9,
                                             weight_decay=0.0005)
       
        
        trained_model.auto_train(epochs=10, max_deviations=3)
        