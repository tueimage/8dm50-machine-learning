# Extra functions Group 11

def random_brightness_augmentation(patch):
    """ 
    Adjust brightness by adding a random offset to the image intensity.
    
    :param patch: input image patch with values in the range [0, 1]
    """
    # Smaller random brightness offset in the range [-0.3, 0.3]
    random_brightness_offset = np.random.uniform(-0.3, 0.3, size=(1, 1, 1))
    
    # Apply brightness adjustment
    patch[..., :3] += random_brightness_offset
    
    # Clip values to stay in the range [0, 1]
    patch = np.clip(patch, 0, 1)
    
    return patch

def random_bspline(patch, seg):
    """
    Applies geometric augmentation to an image/patch. The output is an altered path and it's corresponding segmentation.
    """
    # Assuming image has the shape (584, 565, 3)
    # Define a random 3x3 B-spline grid for a 2D image:

    random_grid = np.random.rand(2, 3, 3)  # 2D grid: 2 displacement directions (x, y) for a 3x3 control points grid
    random_grid -= 0.5  # Center around zero
    random_grid /= 5    # Scale down for small displacements

    # Define a 2D B-spline transformation object using the random grid
    bspline = gryds.BSplineTransformation(random_grid)

    # List to hold transformed channels
    channels = []

    # Loop over the 3 color channels (R, G, B)
    for i in range(3):
        # Create an interpolator for the current channel (2D)
        channel_interpolator = gryds.Interpolator(patch[:, :, i])
            
        # Transform the current channel using the 2D B-spline transformation
        transformed_channel = channel_interpolator.transform(bspline)
            
        # Append the transformed channel to the list
        channels.append(transformed_channel)

    # Stack the transformed channels back together into a single image
    new_patch = np.stack(channels, axis=-1)  # Stack along the last axis (color channels)
        
    # Now, transform the segmentation (patch_segmentation) using nearest-neighbor interpolation
    segmentation_interpolator = gryds.Interpolator(seg[:, :, 0], order=0)  # Nearest-neighbor for segmentation
    new_seg = segmentation_interpolator.transform(bspline)
    return new_patch, new_seg

def alter_patch(patches, patches_segmentations, bspline=False, brightness=False, nr_copies=1):
    """
    Alters patches (and patch segmentations) a number of times (depending on nr_copies) to produce new data.
    """
    patches_og = patches
    patches_final = patches

    patches_segmentations_og = patches_segmentations
    patches_segmentations_final = patches_segmentations

    patches_brightness = patches
    patches_segmentations_brightness = patches_segmentations

    if brightness:
        for i in range(patches_brightness.shape[0]):
            patches_brightness[i, :, :, :] = random_brightness_augmentation(patches_og[i])
            patches_segmentations_brightness[i, :, :, :] = patches_segmentations_og[i]
        
        if not bspline:
            patches_final = np.concatenate((patches_brightness, patches_og), axis=0)
            patches_segmentations_final = np.concatenate((patches_segmentations_brightness, patches_segmentations_og), axis=0)
        else:
            for i in range(patches_og.shape[0]):
                for j in range(nr_copies):
                    new_patch, new_seg = random_bspline(patches_brightness[i], patches_segmentations_og[i])
                    
                    # Expanding matrix to combine with original patches and segmentations
                    new_patch_dim = np.expand_dims(new_patch, axis=0)
                    new_seg_dim = np.expand_dims(new_seg, axis=0)
                    new_seg_dim_2 = np.expand_dims(new_seg_dim, axis=-1)

                    # Clip new_patch to ensure valid RGB range
                    new_patch_dim = np.clip(new_patch_dim, 0, 1)

                    # Append the new matrix to the existing tensor
                    patches_final = np.concatenate((patches_final, new_patch_dim), axis=0)
                    patches_segmentations_final = np.concatenate((patches_segmentations_final, new_seg_dim_2), axis=0)
  
    return patches_final, patches_segmentations_final

# Create a very simple datagenerator
def datagenerator1(images, segmentations, patch_size, patches_per_im, batch_size, brightness_augmentation= False, bspline_augmentation= False):
    """
    Altered datagenerator. Simple data-generator to feed patches in batches to the network.
    To extract different patches each epoch, steps_per_epoch in fit_generator should be equal to nr_batches.
    Includes the ability to apply brightness and bspline augmentation to the extracted patches.

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
        patches, patches_segmentations = extract_patches(images, segmentations, patch_size, patches_per_im, seed=7)
        new_patches, new_segmentations = alter_patch(patches, patches_segmentations, brightness = brightness_augmentation, bspline = bspline_augmentation,  nr_copies = 1)

        # Feed data in batches to the network
        for idx in range(nr_batches):
            x_batch = new_patches[idx * batch_size:(idx + 1) * batch_size]
            y_batch = new_segmentations[idx * batch_size:(idx + 1) * batch_size]
            yield x_batch, y_batch
