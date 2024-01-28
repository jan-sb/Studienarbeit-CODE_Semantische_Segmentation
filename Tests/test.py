import torch
import numpy as np
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt




ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = "Studienarbeit-CODE_Semantische_Segmentation/testdata/inst_picset/frame_1038.png"
input_image = Image.open(image_path).convert("RGB")  
input_tensor = transform(input_image)  
input_batch = input_tensor.unsqueeze(0)  


with torch.no_grad():
    output = model(input_batch)['out'][0]  
output_predictions = output.argmax(0) 

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')


# Plot the segmentation result
plt.subplot(1, 2, 2)
plt.imshow(output_predictions, cmap='viridis')
plt.title('Semantic Segmentation')
plt.savefig(f"Studienarbeit-CODE_Semantische_Segmentation/Ergebnisse/testdata/FPN.png")