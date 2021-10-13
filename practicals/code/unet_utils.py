import numpy as np
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
import gryds
import time
import matplotlib.pyplot as plt
import tifffile as tif # ADDED TO WORK ON WINDOWS
from random import randrange
import gryds


def load_data(impaths_all, test=False):
    """
    Load data with corresponding masks and segmentations

    :param impaths_all: Paths of images to be loaded
    :param test: Boolean, part of test set?
    :return: Numpy array of images, masks and segmentations
    """
    # Save all images, masks and segmentations
    images = []
    masks = []
    segmentations = []

    # Load as numpy array and normalize between 0 and 1
    for im_path in impaths_all:
        #images.append(np.array(Image.open(im_path)) / 255.) # UNCOMMENT TO RUN ON LINUX
        images.append(np.array(tif.imread(im_path)) / 255.) # ADDED TO WORK ON WINDOWS, COMMENT TO RUN ON LINUX
        mask_path = im_path.replace('images', 'mask').replace('.tif', '_mask.gif')
        masks.append(np.array(Image.open(mask_path)) / 255.)
        if not test:
            seg_path = im_path.replace('images', '1st_manual').replace('training.tif', 'manual1.gif')
        else:
            seg_path = im_path.replace('images', '1st_manual').replace('test.tif', 'manual1.gif')
        segmentations.append(np.array(Image.open(seg_path)) / 255.)

    # Convert to numpy arrays with channels last and return
    return np.array(images), np.expand_dims(np.array(masks), axis=-1), np.expand_dims(np.array(segmentations), axis=-1)


def pad_image(image, desired_shape):
    """
    Pad image to square

    :param image: Input image
    :param desired_shape: Desired shape of padded image
    :return: Padded image
    """
    padded_image = np.zeros((desired_shape[0], desired_shape[1], image.shape[-1]), dtype=image.dtype)
    pad_val_x = desired_shape[0] - image.shape[0]
    pad_val_y = desired_shape[1] - image.shape[1]
    padded_image[int(np.ceil(pad_val_x / 2)):padded_image.shape[0]-int(np.floor(pad_val_x / 2)),
                 int(np.ceil(pad_val_y / 2)):padded_image.shape[0]-int(np.floor(pad_val_y / 2)), :] = image
    return padded_image


# Pad to squares
def preprocessing(images, masks, segmentations, desired_shape):
    """
    Pad all images, masks and segmentations to desired shape

    :param images: Numpy array of images
    :param masks: Numpy array of masks
    :param segmentations: Numpy array of segmentations
    :param desired_shape: Desired shape of padded image
    :return: Padded images, masks and segmentations
    """
    padded_images = []
    padded_masks = []
    padded_segmentations = []
    for im, mask, seg in zip(images, masks, segmentations):
        padded_images.append(pad_image(im, desired_shape))
        padded_masks.append(pad_image(mask, desired_shape))
        padded_segmentations.append(pad_image(seg, desired_shape))

    return np.array(padded_images), np.array(padded_masks), np.array(padded_segmentations)


def extract_patches(images, segmentations, patch_size, patches_per_im, seed):
    """
    Extract patches from images

    :param images: Input images
    :param segmentations: Corresponding segmentations
    :param patch_size: Desired patch size
    :param patches_per_im: Amount of patches to extract per image
    :param seed: Random seed to ensure matching patches between image and segmentation
    :return: x: numpy array of patches and y: numpy array of patches segmentations
    """
    # The total amount of patches that will be obtained
    inp_size = len(images) * patches_per_im
    # Allocate memory for the patches and segmentations of the patches
    x = np.zeros((inp_size, patch_size[0], patch_size[1], images.shape[-1]))
    y = np.zeros((inp_size, patch_size[0], patch_size[1], segmentations.shape[-1]))

    # Loop over all the images (and corresponding segmentations) and extract random patches 
    # using the extract_patches_2d function of scikit learn
    for idx, (im, seg) in enumerate(zip(images, segmentations)):
        # Note the random seed to ensure the corresponding segmentation is extracted for each patch
        x[idx * patches_per_im:(idx + 1) * patches_per_im] = extract_patches_2d(im, patch_size,
                                                                                max_patches=patches_per_im,
                                                                                random_state=seed)
        y[idx * patches_per_im:(idx + 1) * patches_per_im] = np.expand_dims(
            extract_patches_2d(seg, patch_size, max_patches=patches_per_im, random_state=seed),
            axis=-1)

    return x, y


# Create a very simple datagenerator
def datagenerator(images, segmentations, patch_size, patches_per_im, batch_size):
    """
    Simple data-generator to feed patches in batches to the network.
    To extract different patches each epoch, steps_per_epoch in fit_generator should be equal to nr_batches.

    :param images: Input images
    :param segmentations: Corresponding segmentations
    :param patch_size: Desired patch size
    :param patches_per_im: Amount of patches to extract per image
    :param batch_size: Number of patches per batch
    :return: Batch of patches to feed to the model
    """
    # Total number of patches generated per epoch
    total_patches = len(images) * patches_per_im
    # Amount of batches in one epoch
    nr_batches = int(np.ceil(total_patches / batch_size))

    while True:
        # Each epoch extract different patches from the training images
        x, y = extract_patches(images, segmentations, patch_size, patches_per_im, seed=np.random.randint(0, 500))

        # Feed data in batches to the network
        for idx in range(nr_batches):
            x_batch = x[idx * batch_size:(idx + 1) * batch_size]
            y_batch = y[idx * batch_size:(idx + 1) * batch_size]
            yield x_batch, y_batch



def data_aug(x, y, z, aug_number, min_offset, max_offset, bspline):
    """
    :param x: Input images
    :param y: Corresponding segmentations
    :param z: Corresponding masks
    :param aug_number: Number of augmentations to be performed
    :params min_offset and max_offset: the brightness offset applied to the image is a random float between min_offset and max_offset
    :param bspline: True if we want to perform a b-spline geometric augmentation in addition to brightness augmentation, False if we only want to perform brightness augmentation.
    
    Output: the function data_aug returns: 
    - original_new_x: input images + augmented images
    - original_new_y: input segmentations + augmented segmentations
    - original_new_z: input masks + augmented masks
    """
    new_x = np.zeros((aug_number, x.shape[1], x.shape[2], x.shape[3]))
    new_y = np.zeros((aug_number, y.shape[1], y.shape[2], y.shape[3]))
    new_z = np.zeros((aug_number, z.shape[1], z.shape[2], z.shape[3]))

    for i in range(0, aug_number, 1):   
        
        im_nr = randrange(len(x)) #random image to perform data augmentation
        
        brightness = np.random.uniform(min_offset, max_offset)
        x_random_brightness = np.array(brightness+x[im_nr])
        #y_random_brightness = np.array(brightness+y[im_nr])
        #z_random_brightness = np.array(brightness+z[im_nr])
        x_random_brightness[x_random_brightness > 1] = 1 # clip data to valid range ([0..1])
        #y_random_brightness[y_random_brightness > 1] = 1 # clip data to valid range ([0..1])
        #z_random_brightness[z_random_brightness > 1] = 1 # clip data to valid range ([0..1])
        x_random_brightness[x_random_brightness < 0] = 0 # clip data to valid range ([0..1])
        #y_random_brightness[y_random_brightness < 0] = 0 # clip data to valid range ([0..1])
        #z_random_brightness[z_random_brightness < 0] = 0 # clip data to valid range ([0..1])
            
        if bspline:  
            transformed = np.zeros_like(x[im_nr])  
            
            # Define a random 3x3 B-spline grid for a 2D image:
            random_grid = np.random.rand(2,3,3)
            random_grid -= 0.5
            random_grid /= 5

            # Define a B-spline transformation object
            bspline = gryds.BSplineTransformation(random_grid)
            
            # Define an interpolator object for the image:
            # Images x are RGB images - they have three color channels (3D), so we have to separate them in orer to define the interpolators.
            interpolator_x1 = gryds.Interpolator(x[im_nr,:,:,0])
            interpolator_x2 = gryds.Interpolator(x[im_nr,:,:,1])
            interpolator_x3 = gryds.Interpolator(x[im_nr,:,:,2])
            interpolator_y = gryds.Interpolator(y[im_nr,:,:,0])
            interpolator_z = gryds.Interpolator(z[im_nr,:,:,0])
                
            transformed_image_x1 = interpolator_x1.transform(bspline)
            transformed_image_x2 = interpolator_x2.transform(bspline)
            transformed_image_x3 = interpolator_x3.transform(bspline)
            transformed_image_y = interpolator_y.transform(bspline)
            transformed_image_z = interpolator_z.transform(bspline)
            
            # Join the three color channels together            
            transformed[:,:,0], transformed[:,:,1], transformed[:,:,2] = transformed_image_x1, transformed_image_x2, transformed_image_x3
            
            new_x[i, :, :, :] = transformed
            new_y[i, :, :, :] = transformed_image_y[:, :, np.newaxis]
            new_z[i, :, :, :] = transformed_image_z[:, :, np.newaxis]
            
            original_new_x = np.concatenate((x, new_x), axis = 0)
            original_new_y = np.concatenate((y, new_y), axis = 0)
            original_new_z = np.concatenate((z, new_z), axis = 0)
        
        else:
            new_x[i, :, :, :] = x_random_brightness
            new_y[i, :, :, :] = y_random_brightness
            new_z[i, :, :, :] = z_random_brightness
            
            original_new_x = np.concatenate((x, new_x), axis = 0)
            original_new_y = np.concatenate((y, new_y), axis = 0)
            original_new_z = np.concatenate((z, new_z), axis = 0)
            
    return original_new_x, original_new_y, original_new_z