import torch
import numpy as np
from torchvision import transforms
import torchseg as ts
from PIL import Image
import matplotlib.pyplot as plt




model = ts.DeepLabV3Plus(
    encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights= "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=21,                      # model output channels (number of classes in your dataset)
)

model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Pad(padding=(0,4)),    
])

image_path = "Studienarbeit-CODE_Semantische_Segmentation/testdata/inst_picset/frame_1038.png" #1038
input_image = Image.open(image_path).convert("RGB")  
input_tensor = transform(input_image)  
input_batch = input_tensor.unsqueeze(0)  




with torch.no_grad():
    output = model.forward(input_batch)
print(output.shape)
output_predictions = output.argmax(1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')


# Plot the segmentation result
plt.subplot(1, 2, 2)
plt.imshow(output_predictions[0], cmap='viridis')
plt.title('Semantic Segmentation')
plt.savefig(f"Studienarbeit-CODE_Semantische_Segmentation/Ergebnisse/ergdata/latest.png")