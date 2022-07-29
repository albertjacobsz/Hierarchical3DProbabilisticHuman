"""
This python file is used to create new images out of two input images
These input images consist of an overlay image and a background image
The overlay image is an image of a human being which has a particular pose (e,g. walking towards the right, left, etc.)
The background image, as suggested, is the background image over which the overlay image is put

REQUIREMENTS FOR OVERLAY IMAGE AND BACKGROUND IMAGE
1. The overlay image should only be of a human being,
this means that there should be no visible background 
although some white pixels should be fine
2. The background images should not have any humans in the picture as it could disturb with the model


"""
import torch
from PIL import Image, ImageDraw,ImageFilter
import os
from random import seed
from random import randint

def main():
    path_background = './input_images/background_images/'
    path_overlay = './input_images/overlay_img/'
    seed(1)
    background_dirlist = os.listdir(path_background)
    overlay_dirlist = os.listdir(path_overlay)
    for ovly in overlay_dirlist:
        ovly_img = Image.open(path_overlay+ovly)
        for bkgnd in background_dirlist:
            bkgnd_img = Image.open(path_background+bkgnd)
            # Convert image to RGBA
            ovly_img = ovly_img.convert("RGBA")
            bkgnd_img = bkgnd_img.convert("RGBA")
            width, height = bkgnd_img.size
            for i in range(0,2):
                rt = ovly_img.rotate(randint(0,360),expand=1)
                merge_img = bkgnd_img.copy()
                x =int(randint(0,width)/2)
                y =int(randint(0,height)/2)
                merge_img.paste(rt,(x,y),rt)
                merge_img.save('./demo/'+ovly.split(".")[0]+'_'+bkgnd.split('.')[0]+'_'+str(x)+'_'+str(y)+'_'+str(i)+'.png',format="png")
if __name__ == "__main__":
    main()

