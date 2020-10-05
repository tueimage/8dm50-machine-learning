# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 01:40:46 2020

@author: s140468
"""
import tensorflow as tf
import gryds
import numpy as np


def brightness_aug(imgs,max_delta):
    return [tf.compat.v1.image.random_brightness(img,max_delta) for img in imgs]

# def bspline_aug(imgs,disp_i,disp_j):
#     import gryds
#     return [gryds.MultiChannelInterpolator(img, order=0, cval=[.1, .2, .3]).transform(gryds.BSplineTransformation([disp_i, disp_j])) for img in imgs]

def bspline_aug(imgs, segs):
    
    import numpy as np
    import sys
    sys.path.append('./gryds/')
    import gryds
    augmented =[]
    segmented =[]
    for i in range(len(imgs)):
        random_grid = np.random.rand(2, 3, 3) # Make a random 2D 3 x 3 grid
        random_grid-= 0.5 # Move the displacements to the -0.5 to 0.5 grid
        random_grid /= 10 # Scale the grid to -0.1 to 0.1 displacements
        
        the_image_interpolator = gryds.MultiChannelInterpolator(imgs[i], order=0, cval=[.1, .2, .3])
        the_augmentation = gryds.BSplineTransformation(random_grid)
        the_augmented_image = the_image_interpolator.transform(the_augmentation)
        augmented.append(the_augmented_image)

        the_segment_interpolator = gryds.Interpolator(segs[i][:,:,0], mode='constant')
        the_augmentation_segment = gryds.BSplineTransformation(random_grid)
        the_segmented_image = the_segment_interpolator.transform(the_augmentation_segment)
        segmented.append(the_segmented_image)
    return augmented,segmented