import torch
import torchvision.transforms.functional as tvf
from PIL import Image as PImage
from PIL import ImageOps
# import matplotlib.pyplot as plt
# import numpy as np

def _resize_pad_pil(pilImage, desired_size, mode='RGB'):
    # if not isinstance(desired_size, tuple):
        # desired_size = (desired_size, desired_size)

    old_size = pilImage.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(sz*ratio) for sz in old_size])
    pilImage = pilImage.resize(new_size, PImage.ANTIALIAS)
    
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, 
                delta_w-(delta_w//2), 
                delta_h-(delta_h//2))
    
    return ImageOps.expand(pilImage, padding)

class ResizePadPilImage(object):
    """
    Assuming the torch tensor is channel last.
    This class resizes the image to a image of size 'desired_size'.
    It maintains the aspect ratio and pads the final image with zeros.

    TODO: 
        1. Add ability to scale to non-square image with padding
    """
    def __init__(self, desired_size, mode='RGB'):
        self.desired_size = desired_size
        self.mode = mode
    
    def __call__(self, x, y=None):
        if x.mode != self.mode:
           x = x.convert(self.mode)
        x = tvf.to_tensor(_resize_pad_pil(x, self.desired_size))
        # x = tvf.to_tensor(x)
        if y is not None:
            y = tvf.to_tensor(_resize_pad_pil(y, self.desired_size))
            # y = tvf.to_tensor(y)
            return x, y
        else:
            #plt.imshow(np.rollaxis(x.numpy(), 0, 3))
            #plt.show()
            return x
        
        

