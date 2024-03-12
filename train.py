import os
import cv2 as cv
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as transf

learning_rate = 1 * 10**(-5)


