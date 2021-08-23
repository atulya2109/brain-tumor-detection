import os
import skimage.io as io
import numpy as np
from skimage.transform import rotate
from skimage.util import img_as_ubyte
import argparse

def augment_images(path):
    
    generated_imgs = 0
    imgs_augmntd = 0
    for img in os.listdir(path):
        try:
            image = io.imread(f'{os.path.join(path,img)}')

            #rotate images upto 360 degree

            for c in range(1,360,2):
                rotated = rotate(image,angle=c,mode='wrap')
                rotated_name = f'{img[:img.rfind(".")]}_r{c}'
                rotated_path =  os.path.join(path,f'{rotated_name}.jpg')
                io.imsave(fname = rotated_path,arr = img_as_ubyte(rotated))

                flipLR = np.fliplr(image)
                flipped_name = f'{rotated_name}_LR'
                flipped_path = os.path.join(path,f'{flipped_name}.jpg')
                io.imsave(fname = flipped_path,arr = img_as_ubyte(flipLR))

                flipUD = np.flipud(image)
                flipped_name = f'{rotated_name}_UD'
                flipped_path = os.path.join(path,f'{flipped_name}.jpg')
                io.imsave(fname = flipped_path, arr = img_as_ubyte(flipUD))

                generated_imgs += 3

                print(f'Images Augmented: {imgs_augmntd} | Images Generated: {generated_imgs}', end='\r')
        except:
            pass
            
        imgs_augmntd += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    augment_images(args.path)