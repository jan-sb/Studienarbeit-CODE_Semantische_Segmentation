from Helper.ml_models import *
from Helper.Helper_functions import * 
from torchvision.models.segmentation import *
import gc, sys

deeplv3 = ['deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet50']
fcn = ['fcn_resnet50', 'fcn_resnet101']
lrsapp = ['lraspp_mobilenet_v3_large']

all_models = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'fcn_resnet50', 'fcn_resnet101', 'lraspp_mobilenet_v3_large']


##### UNCOMMENT before Big RUN
# for model in all_models:
#     for i in range(5):
#         create_model_directory(model, i)
#         trained_model = TrainedModel(model, 2048, 1024, f'{model}_k_fold_{i}', start_epoch='latest')
#         del trained_model
#         gc.collect()
# # sys.exit()

total_eppochs = 15
epoch_steps = 1
runs = total_eppochs // epoch_steps



for model in all_models:
    try: 
        for i in range(5):
            i = 0
            create_model_directory(model, i)
            trained_model = TrainedModel(model, 2048, 1024, f'{model}_k_fold_{i}', start_epoch='latest')
            k_fold_dataset = K_Fold_Dataset(image_dir='CityscapesDaten/images',
                                            annotation_dir='CityscapesDaten/semantic',
                                            k_fold_csv_dir='Daten/CityscapesDaten',
                                            leave_out_fold=i)        
            
            k_fold_dataset.check_for_data_leaks()               
            
            trained_model.prepare_model_training(dataset_train=k_fold_dataset.train_dataset,
                                                dataset_val=k_fold_dataset.val_dataset,
                                                batch_size=2, 
                                                shuffle=True, 
                                                learning_rate=1*10**(-7), 
                                                momentum=0.9,
                                                weight_decay=0.00001)
        
            
            trained_model.auto_train(epochs=total_eppochs, max_deviations=5)
            #trained_model.train(1)
            
            # image, annotation = k_fold_dataset.train_dataset[0]
            
            # visualize_image_and_annotation(image, annotation)
            
                
            trained_model.writer.flush()
            trained_model.writer.close()
    except Exception as e:
        error_message = str(e)
        with open('error.json', 'w') as f:
            json.dump({'error': error_message}, f)
        os.system('sudo shutdown now -h')