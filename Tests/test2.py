import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


weights = 'DEFAULT'

# Load the pretrained DeepLabV3 model with a ResNet backbone
model = models.segmentation.fcn_resnet50(weights=weights, num_classes= 21)
model.eval()  # Set the model to evaluation mode (no gradient computation during inference)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
])

# Load and preprocess the input image
image_path = "Studienarbeit-CODE_Semantische_Segmentation/testdata/inst_picset/frame_1038.png"
input_image = Image.open(image_path).convert("RGB")  # Open the image in RGB mode
input_tensor = transform(input_image)  # Apply the defined transformations
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension (single image)

# Perform inference
with torch.no_grad():
    output = model(input_batch)['out'][0]  # Forward pass through the model
output_predictions = output.argmax(0)  # Get the index of the most probable class for each pixel

# Visualize the original image and segmentation result
plt.figure(figsize=(12, 6))
# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')

# Plot the segmentation result
plt.subplot(1, 2, 2)
plt.imshow(output_predictions, cmap='viridis')
plt.title('Semantic Segmentation')
plt.savefig(f"Studienarbeit-CODE_Semantische_Segmentation/Ergebnisse/testdata/FCN50_{weights}.png")
