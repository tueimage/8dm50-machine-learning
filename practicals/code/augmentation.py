# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 01:40:46 2020

@author: s140468
"""
import tensorflow as tf
import gryds

def brightness_aug(imgs,max_delta):
    return [tf.compat.v1.image.random_brightness(img,max_delta) for img in imgs]

def bspline_aug(imgs,disp_i,disp_j):
    import gryds
    return [gryds.MultiChannelInterpolator(img, order=0, cval=[.1, .2, .3]).transform(gryds.BSplineTransformation([disp_i, disp_j])) for img in imgs]