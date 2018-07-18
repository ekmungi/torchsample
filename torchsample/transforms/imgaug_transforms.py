import numpy as np
import torch
import cv2
from imgaug import augmenters as iaa


# class AugmentorBaseClass(object):
#     """
#     Augmentor base class with no implementations
#     """
#     def __init__(self, desired_size, mode='RGB'):
    
#     def __call__(self):
#         NotImplementedError()

def _border(border):
    if isinstance(border, tuple):
        if len(border) == 2:
            left, top = right, bottom = border
        elif len(border) == 4:
            left, top, right, bottom = border
    else:
        left = top = right = bottom = border
    return left, top, right, bottom

def _resize_pad_array(x, desired_size):
    # if not isinstance(desired_size, tuple):
        # desired_size = (desired_size, desired_size)

    old_size = x.shape
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(sz*ratio) for sz in old_size])
    x = cv2.resize(x.astype(np.float32), (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    # return ImageOps.expand(x, padding)
    return cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

class ResizePadArray(object):
    """
    This class resizes the image to a image of size 'desired_size'.
    It maintains the aspect ratio and pads the final image with zeros.
    Accepts an image array or torch tensor

    TODO: 
        1. Add ability to scale to non-square image with padding
    """
    def __init__(self, desired_size):
        self.desired_size = desired_size
    
    def __call__(self, x, y=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(_resize_pad_array(x, self.desired_size))
        elif isinstance(x, torch.Tensor):
            x = torch.from_numpy(_resize_pad_array(x.numpy(), self.desired_size))
        if y is not None:
            y = torch.from_numpy(_resize_pad_array(y, self.desired_size))
            return x, y
        else:
            return x

class ExpandDims(object):
    """
    Add additional dimension to image
    """
    def __init__(self, axis):
        self.axis = axis
    
    def __call__(self, x, y=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.expand_dims(x, axis=self.axis))
        elif isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.expand_dims(x.numpy(), axis=self.axis))
        if y is not None:
            y = torch.from_numpy(_resize_pad_array(y, self.desired_size))
            return x, y
        else:
            return x

def _augmentImgLabelNumpy(images, labels, transform_list):
    # augmenters_imgs = [
    # iaa.Affine(        
    #     rotate=(-55, 35)
    # )]         
                      
    seq_transforms = iaa.Sequential(transform_list, random_order=False)        
    seq_transforms_deterministic = seq_transforms.to_deterministic()
    
    images_aug = seq_transforms_deterministic.augment_images(images)
    masks_aug = seq_transforms_deterministic.augment_images(labels)
    return (images_aug, masks_aug)

class ImgAugCoTranform(object):
    """
    Wrapper for transforms from imgaug to both images and labels 
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list
    
    def __call__(self, x, y=None):
        
        if isinstance(x, np.ndarray):
            x, y = _augmentImgLabelNumpy(np.rollaxis(x, 1, 4), np.rollaxis(y, 1, 4), self.transform_list)
            return (torch.from_numpy(np.rollaxis(x, 3, 1)), torch.from_numpy(np.rollaxis(y, 3, 1)))
        if isinstance(x, torch.Tensor):
            x, y = _augmentImgLabelNumpy(np.rollaxis(x.numpy(), 1, 4), np.rollaxis(y.numpy(), 1, 4), 
                                            self.transform_list)
            return (torch.from_numpy(np.rollaxis(x, 3, 1)), torch.from_numpy(np.rollaxis(y, 3, 1)))
            

def _augmentNumpy(x, transform_list):
    # augmenters_imgs = [
    # iaa.Affine(        
    #     rotate=(-55, 35)
    # )]                           
    seq_transforms = iaa.Sequential(transform_list, random_order=False)        
    # seq_imgs_deterministic = augment_seq.to_deterministic()
    # masks_aug = seq_imgs_deterministic.augment_images(labels)
    return seq_transforms.augment_images(x)

class ImgAugTranform(object):
    """
    Wrapper for transforms from imgaug to images only
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list
    
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.rollaxis(_augmentNumpy(np.rollaxis(x, 1, 4), 
                                    self.transform_list), 3, 1))
        elif isinstance(x, torch.Tensor):
            return torch.from_numpy(np.rollaxis(_augmentNumpy(np.rollaxis(x.numpy(), 1, 4), 
                                    self.transform_list), 3, 1))