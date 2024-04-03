import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import model  # Ensure this module contains the necessary functions for the GAN
import argparse
from os.path import join
import imageio
import h5py
import os


def main():
    parser = argparse.ArgumentParser()

    # Updated default value for caption_vector_length to match the actual data
    parser.add_argument('--caption_vector_length', type=int, default=768, help='Caption Vector Length')

    # Other arguments remain the same
    parser.add_argument('--z_dim', type=int, default=100, help='Noise Dimension')
    parser.add_argument('--t_dim', type=int, default=256, help='Text feature dimension')
    parser.add_argument('--image_size', type=int, default=64, help='Image Size')
    parser.add_argument('--gf_dim', type=int, default=64, help='Number of conv in the first layer gen.')
    parser.add_argument('--df_dim', type=int, default=64, help='Number of conv in the first layer discr.')
    parser.add_argument('--gfc_dim', type=int, default=1024,help='Dimension of gen units for fully connected layer 1024')
    parser.add_argument('--data_dir', type=str, default="Data", help='Data Directory')
    parser.add_argument('--model_path', type=str, default='Data/latest_model_flowers_temp.ckpt',help='Trained Model Path')
    parser.add_argument('--n_images', type=int, default=5, help='Number of Images per Caption')
    parser.add_argument('--caption_thought_vectors', type=str, default='Data/sample_caption_vectors.hdf5',
                        help='Caption Thought Vector File')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizer')
    parser.add_argument('--resume_model', type=str, default=None,
                        help='Path to a pre-trained model to resume training from')

    args = parser.parse_args()

    model_options = {
        'z_dim': args.z_dim,
        't_dim': args.t_dim,
        'batch_size': args.n_images,
        'image_size': args.image_size,
        'gf_dim': args.gf_dim,
        'df_dim': args.df_dim,
        'gfc_dim': args.gfc_dim,
        'caption_vector_length': args.caption_vector_length  # This will now use the updated default value
    }

    gan = model.GAN(model_options)
    input_tensors, variables, loss, outputs, checks = gan.build_model()

    # Optimizers
    tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
    tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    input_tensors, outputs = gan.build_generator()

    h = h5py.File(args.caption_thought_vectors, 'r')
    caption_vectors = np.array(h['vectors'])
    h.close()

    caption_image_dic = {}
    for cn, caption_vector in enumerate(caption_vectors):
        caption_images = []
        z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])

        # Now the caption vectors match the expected shape
        caption = [caption_vector] * args.n_images

        [gen_image] = sess.run([outputs['generator']],
                               feed_dict={
                                   input_tensors['t_real_caption']: caption,
                                   input_tensors['t_z']: z_noise,
                               })

        caption_images = [gen_image[i, :, :, :] for i in range(args.n_images)]
        caption_image_dic[cn] = caption_images
        print("Generated", cn)

    for f in os.listdir(join(args.data_dir, 'val_samples')):
        if os.path.isfile(f):
            os.unlink(join(args.data_dir, 'val_samples/' + f))

    for cn in range(0, len(caption_vectors)):
        caption_images = []
        for i, im in enumerate(caption_image_dic[cn]):
            im_name = "caption_{}_{}.jpg".format(cn, i)
            # Scale the image data to 0-255 and convert to uint8
            im = (im * 255).astype(np.uint8)
            imageio.imwrite(join(args.data_dir, 'val_samples/{}'.format(im_name)), im)
            caption_images.append(im)
            caption_images.append(np.zeros((64, 5, 3), dtype=np.uint8))
        combined_image = np.concatenate(caption_images[0:-1], axis=1)
        # Also scale and convert the combined image
        combined_image = (combined_image * 255).astype(np.uint8)
        imageio.imwrite(join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn)), combined_image)


if __name__ == '__main__':
    main()