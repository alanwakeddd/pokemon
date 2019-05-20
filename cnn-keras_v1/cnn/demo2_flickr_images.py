import flickrapi
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import requests
from io import BytesIO
import os
import warnings

api_key = u'45907175c467a1e968e47ab65ac17f7d'
api_secret = u'xxxxxxxxxxxxxxxx'
flickr = flickrapi.FlickrAPI(api_key, api_secret)
#'Eevee', 'Squirtle', 'Charmander', 'Umbereon', 'Arcanine', 'Charizard',
keyword_list = ['Mudkip', 'Lucario', 'Bulbasaur', 'Jigglypuff']


# Display the image
def disp_image(im):
    if (len(im.shape) == 2):
        # Gray scale image
        plt.imshow(im, cmap='gray')    
    else:
        # Color image.  
        im1 = (im-np.min(im))/(np.max(im)-np.min(im))*255
        im1 = im1.astype(np.uint8)
        plt.imshow(im1)    
        
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])


for key in keyword_list:
    keyword = key
    dir_name = key
    photos = flickr.walk(text=keyword, tag_mode='all', tags=keyword, extras='url_c', sort='relevance', per_page=100)
    dir_exists = os.path.isdir(dir_name)
    if not dir_exists:
        os.makedirs(dir_name)
        print("Making directory %s" % dir_name)
    else:
        print("Will store images in directory %s" % dir_name)

    nimage = 1000
    i = 0
    nrow = 224
    ncol = 224
    for photo in photos:
        url = photo.get('url_c')
        if not (url is None):
            # Create a file from the URL
            # This may only work in Python3
            response = requests.get(url)
            file = BytesIO(response.content)

            # Read image from file
            im = skimage.io.imread(file)

            # Resize images
            im1 = skimage.transform.resize(im, (nrow, ncol), mode='constant')

            # Convert to uint8, suppress the warning about the precision loss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im2 = skimage.img_as_ubyte(im1)

            # Save the image
            local_name = '{0:s}/{1:s}_{2:04d}.jpg'.format(dir_name, keyword, i)
            skimage.io.imsave(local_name, im2)
            print(local_name)
            i = i + 1
        if (i >= nimage):
            break

if 0:
    plt.figure(figsize=(20, 20))
    nplot = 4
    for i in range(nplot):
        fn = '{0:s}/{1:s}_{2:04d}.jpg'.format(keyword, keyword, i)
        im = skimage.io.imread(fn)
        plt.subplot(1, nplot, i + 1)
        disp_image(im)






