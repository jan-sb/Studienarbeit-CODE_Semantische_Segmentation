from Helper.ml_models import *




def main():

    '''hybrid_model = hub_model('datvuthanh/hybridnets',
                             'hybridnets',
                             pretrained=True)'''

    deeplv3 = torch_model('fcn_resnet50',
                          'FCN_ResNet50_Weights',
                          pretrained=True)







if __name__ == '__main__':
    main()