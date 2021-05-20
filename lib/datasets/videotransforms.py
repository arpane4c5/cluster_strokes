import numpy as np
import numbers
import random
from collections.abc import Iterable
from PIL import Image
import torchvision
import torch
from torchvision import transforms

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, c, h, w = img.shape      # when img is (C, H, W)
        th, tw = output_size
        pad_h, pad_w = False, False
        if w < tw:
            tw = w
            pad_w = True
        if h < th:
            th = h
            pad_h = True
        if w == tw and h == th:
            return 0, 0, h, w, pad_h, pad_w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw, pad_h, pad_w

    def __call__(self, imgs):
        
        i, j, h, w, pad_h, pad_w = self.get_params(imgs, self.size)
        
        
        imgs = imgs[:, :, i:i+h, j:j+w]
        if pad_h:
            T, C, H, W = imgs.shape
            pad_width = self.size[0] - H
            imgs_padded = torch.zeros((T, C, H + pad_width, W), dtype=imgs.dtype)
            imgs_padded[:, :, (pad_width//2):H+(pad_width//2)] = imgs
            imgs = imgs_padded
        if pad_w:
            T, C, H, W = imgs.shape
            pad_width = self.size[1] - W
            imgs_padded = torch.zeros((T, C, H, W + pad_width), dtype=imgs.dtype)
            imgs_padded[:, :, :, (pad_width//2):W+(pad_width//2) ] = imgs
            imgs = imgs_padded
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, c, h, w = imgs.shape
        th, tw = self.size
        if h < th:
            pad_width = self.size[0] - h
            imgs_padded = torch.zeros((t, c, h + pad_width, w), dtype=imgs.dtype)
            imgs_padded[:, :, (pad_width//2):h+(pad_width//2) ] = imgs
            imgs = imgs_padded
            t, c, h, w = imgs.shape
        if w < tw:
            pad_width = self.size[1] - w
            imgs_padded = torch.zeros((t, c, h, w + pad_width), dtype=imgs.dtype)
            imgs_padded[:, :, :, (pad_width//2):w+(pad_width//2) ] = imgs
            imgs = imgs_padded
            t, c, h, w = imgs.shape
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, :, i:i+th, j:j+tw]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
#            return np.flip(imgs, axis=2).copy()
            return torch.flip(imgs, [3]).clone()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class ToPILClip(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, clip):
        """
        Args:
            pic (Tensor or numpy.ndarray): Clip to be converted to sequence of PIL Images.

        Returns:
            [PIL Images list]: list of Images converted to PIL Images.
        """
        toPILimage = transforms.ToPILImage(self.mode)
        return [toPILimage(i) for i in clip]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string
    
class Resize(object):
    """Resize the input video clip using PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.NEAREST``
    """

    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        """
        Args:
            clip (PIL Image): Clip to be scaled.

        Returns:
            [list of PIL Images]: Rescaled images list.
        """
        resize_transform = transforms.Resize(self.size, self.interpolation)
        return [resize_transform(i) for i in clip]

    def __repr__(self):
        interpolate_str = transforms.transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class ToTensor(object):
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]
    def __repr__(self):
        return self.__class__.__name__ + '()'

class ScaledNormMinMax(object):
    '''Assumed values are in range [0, 1]. Therefore, always call after ToTensor()
    '''
    def __call__(self, imgmap, minmax=(0, 255)):
        return [ i*(minmax[1]-minmax[0]) + minmax[0] for i in imgmap]
    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToHMDBTensor(object):
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return torch.stack([totensor(i) for i in imgmap])
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for ImageNet
     # given below for Kinetics 400
    def __init__(self, mean = [0.43216, 0.394666, 0.37645], 
                 std = [0.22803, 0.22145, 0.216989]):
        self.mean = mean
        self.std = std
    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]
    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)

class FiveCrop(object):
    '''FiveCrop for video clips. 
    '''
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image or Tensor or list of PIL): Images to be cropped.
        Returns:
            PIL Image: Tuples of Cropped images.
        """
        fiveCrop = transforms.FiveCrop(self.size)
        crops = [fiveCrop(i) for i in imgs]
#        cropped_tups = transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop) \
#                                                                   for crop in crops]))
        return crops        
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    