from PIL import Image
from os import walk, path
from tqdm import tqdm
import numpy as np
import os


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    img = np.array(img).astype('float32')
    img = (img / 127.5) - 1.0
    return img

def unpreprocess_image(img):
    img = (img + 1.0) * 127.5
    img = img.astype('uint8')
    return img

def display_image(image, save_dir = None):
    if image.shape[0] == 1:
        image = image[0]
    image = Image.fromarray(np.clip(image, 0, 255).astype('uint8'))
    image.show()
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        count = 0
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                count += 1
        image.save(os.path.join(root, str(count) + '.png'))



def get_image_file_list(folder_path):
    file_list = []
    for root, dirs, files in walk(folder_path):
        for file in files:
            if file.split('.')[-1] in ['jpeg', 'png', 'jpg', 'bmp', 'tif', 'tiff']:
                file_list.append(path.join(root, file))
    return file_list

def get_image_dataset(file_list):
    img_array = []
    for file in tqdm(file_list):
        img_array.append(load_image(file))
    img_array = np.array(img_array)
    return img_array
