import os
from PIL import Image
import torch
import numpy as np
def tensor2im(input_image, imtype=np.uint8):
    """"
    Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        # get the data from a variable
        if isinstance(input_image, torch.Tensor):  
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()  
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
    else:# if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def mkdirs(paths):
    """create empty directories if they don't exist
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)