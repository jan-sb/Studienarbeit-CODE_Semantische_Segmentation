import torch
import torch.nn.functional as F

def pad_image(image, target_height, target_width):
    # Get the current height and width of the image
    current_height, current_width = image.shape[-2:]

    # Calculate the amount of padding needed for height and width
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)

    # Calculate the padding on each side (top, bottom, left, right)
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Apply padding using F.pad
    padded_image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))

    return padded_image
