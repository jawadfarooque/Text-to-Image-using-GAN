from os.path import join

import numpy as np
from scipy import misc
import random
import skimage
import skimage.io
import skimage.transform


def load_image_array(image_file, image_size):
    img = skimage.io.imread(image_file)
    # GRAYSCALE
    if len(img.shape) == 2:
        img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='uint8')
        img_new[:, :, 0] = img
        img_new[:, :, 1] = img
        img_new[:, :, 2] = img
        img = img_new

    img_resized = skimage.transform.resize(img, (image_size, image_size))

    # FLIP HORIZONTAL WIRH A PROBABILITY 0.5
    if random.random() > 0.5:
        img_resized = np.fliplr(img_resized)

    return img_resized.astype('float32')


def get_training_batch(batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set,
                       loaded_data):
    real_images = np.zeros((batch_size, image_size, image_size, 3))
    wrong_images = np.zeros((batch_size, image_size, image_size, 3))
    captions = np.zeros((batch_size, caption_vector_length))
    z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

    # This is a placeholder for how you might select the images and captions
    for i in range(batch_size):
        idx = (batch_no * batch_size + i) % len(loaded_data['image_list'])
        real_image_file = join(data_dir, data_set, loaded_data['image_list'][idx])
        real_images[i] = load_image_array(real_image_file, image_size)
        # Similarly load wrong_images and captions

    return real_images, wrong_images, captions, z_noise


if __name__ == '__main__':
    # TEST>>>
    arr = load_image_array('Data/samples', 64)
    print(arr.mean())
    # rev = np.fliplr(arr)
    misc.imsave('rev.jpg', arr)