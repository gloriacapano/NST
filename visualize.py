from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 1024 if torch.cuda.is_available() else 128  # use small size if no gpu

# importing vgg-16 pre-trained model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

############## IMAGE #################

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

from io import BytesIO

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def image_resize(image):
    im = Image.open(BytesIO(image))
    
    # Crop the center of the image
    # Get dimensions
    width = im.size[1]  # Get dimensions
    height = im.size[0]
    print(im.size)
    new = min(width, height)
    print(new)
    left = (width - new)/2 
    top = (height - new)/2
    right = (width + new)/2
    bottom = (height + new)/2
    im = im.crop((top, left, bottom, right, )) 

    # Resize the image    
    im = loader(im).unsqueeze(0)
    return im.to(device, torch.float)


# visualizing the content and style images
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated



import sys
def visualize_image(image):
    img = image_resize(image)

    print(style_img.size())

    plt.figure()
    imgshow(img)


if __name__ == '__main__':
     image = sys.argv[0]
     visualize_image(image)
