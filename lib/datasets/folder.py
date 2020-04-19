from PIL import Image

import os
import os.path
import sys


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


#def is_image_file(filename):
#    """Checks if a file is an allowed image extension.
#
#    Args:
#        filename (string): path to a file
#
#    Returns:
#        bool: True if the filename ends with a known image extension
#    """
#    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, train=True, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    if train:
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        images.append(item)
    else:   # for validation or testing sets ( append without the label )
        for fname in sorted(os.listdir(dir)):
            path = os.path.join(dir, fname)
            if is_valid_file(path):
                item = (path, )
                images.append(item)

    return images

def make_strokes_dataset(vids_list, data_path, class_to_idx, train=True, 
                         extensions=None, is_valid_file=None):
    images = []
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    
    for vid in vids_list:
        path = os.path.join(data_path, vid)
        assert os.path.isfile(path), "File does not exist {}".format(path)
        if is_valid_file(path):
            if train:
                item = (path, class_to_idx[list(class_to_idx.keys())[0]])
            else:   # for validation or testing sets ( append without the label )
                item = (path, )
            images.append(item)
        
    return images

#IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


