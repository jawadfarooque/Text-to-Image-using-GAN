Python version 3.8
install tranformers
install Theano 0.8.2
install Tensorflow

1)Run dataloader.py file it will form flower_tv.hdf5 file i.e., a dataset having images and captions
2)In model.py how GAN works, discriminator and generator functions are defined as well
3)To generate images according to your text firstly saved text in a "sample_caption.txt" file,and then Run "generate_thought_vectors.py" file it will make a vector file i.e., sample_caption_vectors.hdf5.
  Then run file "generate_images.py" it will generate images in a file val_samples
4)To train model goto file "train.py" and run this file it will start training the GAN model, by dfault its epochs set to 600, you can also set it according to your need.But its recommended if you want a perfect images train it under 600 epochs

Make sure to start new training of GAN model goto Models folder and delete all the models that stored after some epcohs delete it and start from zero
If you want to change generated image size its dimension then goto "image_processing.py" file and change imagesize: according to your need

 