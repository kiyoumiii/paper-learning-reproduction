from PIL import Image
import numpy as np

def get_greg():
    im_path = 'IMG_20180214_133144.jpg'
    img = Image.open(im_path)
    img = img.resize((816, 612), resample=Image.LANCZOS)
    img = img.crop((108,0,364,256))
    return np.array(img)

if __name__ == '__main__':
    img = get_greg()
    print(img.shape)