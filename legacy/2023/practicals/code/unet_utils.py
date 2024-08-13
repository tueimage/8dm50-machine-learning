import numpy as np
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
import gryds
import time
import matplotlib.pyplot as plt


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
        images.append(np.array(Image.open(im_path)) / 255.)
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

