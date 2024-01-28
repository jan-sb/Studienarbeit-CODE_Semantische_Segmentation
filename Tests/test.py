import torch
import numpy as np
from torchvision import transforms
import torchseg as ts
from PIL import Image
import matplotlib.pyplot as plt



# create segmentation model with pretrained encoder

model = ts.Unet(
    encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=21,                      # model output channels (number of classes in your dataset)
)

model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Pad(padding=(0,4)),    
])

image_path = "Studienarbeit-CODE_Semantische_Segmentation/testdata/inst_picset/frame_1038.png"
input_image = Image.open(image_path).convert("RGB")  
input_tensor = transform(input_image)  
input_batch = input_tensor.unsqueeze(0)  




with torch.no_grad():
    output = model(input_batch)  
output_predictions = output.argmax(1)[0] 

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')


# Plot the segmentation result
plt.subplot(1, 2, 2)
plt.imshow(output_predictions, cmap='viridis')
plt.title('Semantic Segmentation')
plt.savefig(f"Studienarbeit-CODE_Semantische_Segmentation/Ergebnisse/ergdata/UNet_resnet34.png")