import torch
from torchvision import models, transforms, utils, io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_segmentation_masks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

weights = 'DEFAULT'

# Load the pretrained DeepLabV3 model with a ResNet backbone
model = models.segmentation.deeplabv3_resnet101(weights=weights, num_classes= 21)
model.eval()  # Set the model to evaluation mode (no gradient computation during inference)

#Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    #transforms.Resize(520, 520),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
])


#Load and preprocess the input image
image_path = "Studienarbeit-CODE_Semantische_Segmentation/testdata/inst_picset/frame_0005.png"
input_image = Image.open(image_path).convert("RGB")  # Open the image in RGB mode
input_tensor = transform(input_image)  # Apply the defined transformations
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension (single image)

# Perform inference
with torch.no_grad():
    output = model(input_batch)['out']  # Forward pass through the model
print('Shape ', output.shape)
#output_predictions = output.argmax(0)  # Get the index of the most probable class for each pixel

om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
#print ('OM Shape ', om.shape, ' OM Type ', type(om))
#print ('Uniques ' , np.unique(om), ' Max ', max(np.unique(om)))
#print('Unique Number ', len(np.unique(om)))


image = input_image.ToTensor






plt.imshow(om)

plt.savefig('Studienarbeit-CODE_Semantische_Segmentation/Ergebnisse/ergdata/latest.png')


