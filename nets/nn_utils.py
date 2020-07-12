import math
import imageio
import numpy as np
import tensorflow as tf
import constants as const



# import configuration as config
from pydoc import locate
from tensorflow.python.ops import control_flow_ops

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_SCALE_FACTOR = 0.017

def _std_image_normalize(image, stds):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  num_channels = image.get_shape().as_list()[-1]
  if len(stds) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] /= stds[i]
  return tf.concat(axis=3, values=channels)

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=3, values=channels)

def apply_with_random_selector(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge(
        [func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  # print('nn_utils Doing color distort')
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def inception_preprocess(images):
    ## Images are assumed to be [0,255]
    images = tf.to_float(images)
    images = images / 255.0
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)
    return images

def denseNet_preprocess(images):
    ## Images are assumed to be [0,255]
    images = tf.to_float(images)
    images = images / 255.0
    images = _mean_image_subtraction(images, [0.485, 0.456, 0.406])
    images = _std_image_normalize(images, [0.229, 0.224, 0.225])

    return images

def vgg_preprocess(images):
    ## Images are assumed to be [0,255]
    images = tf.to_float(images)
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94


    images = _mean_image_subtraction(images, [_R_MEAN , _G_MEAN, _B_MEAN])


    return images

def cam_vgg_preprocess(image):
    ## Images are assumed to be [0,255]

    # Do nothing already handle it while preparing the crop

    # image = tf.to_float(image)

    # channels = tf.unstack(image, axis=-1)
    # image = tf.stack([channels[2], channels[1], channels[0]], axis=-1) # flip from rgb to bgr
    # image = image -  mean_img # Subtract mean image
    # image = tf.image.resize_bilinear(image , (224,224))
    # _R_MEAN = 123.68
    # _G_MEAN = 116.78
    # _B_MEAN = 103.94
    #
    #
    # image = _mean_image_subtraction(image, [_R_MEAN , _G_MEAN, _B_MEAN])


    return image

def center_crop(images,preprocess_func):
    center_offest = (256 - const.frame_width )//2 # I already resized all images to 256
    images = tf.image.crop_to_bounding_box(images, center_offest , center_offest , const.frame_height, const.frame_width)

    if preprocess_func == 'inception_v1':
        print('Inception Format Augmentation')
        images = inception_preprocess(images)
    elif preprocess_func == 'densenet':
        print('DenseNet Format Augmentation')
        images = denseNet_preprocess(images)
    elif preprocess_func == 'vgg':
        print('VGG Format Augmentation')
        images = vgg_preprocess(images)


    return images

def adjust_color_space(images,preprocess_func):
    if preprocess_func == 'inception_v1':
        print('Inception Format Augmentation')
        images = inception_preprocess(images)
    elif preprocess_func == 'densenet':
        print('DenseNet Format Augmentation')
        images = denseNet_preprocess(images)
    elif preprocess_func == 'vgg':
        print('VGG Format Augmentation')
        images = vgg_preprocess(images)
    elif preprocess_func == 'cam_vgg':
        print('CAM VGG Format Augmentation')
        images = cam_vgg_preprocess(images)
    else:
        raise NotImplementedError('Invalid preprocess_func {}'.format(preprocess_func))

    return images

def augment(images,
            preprocess_func,
            resize=None,  # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0,  # Maximum rotation angle in degrees
            noise_probability = 0,
            color_aug_probability = 0,
            crop_probability=0,  # How often we do crops
            crop_min_percent=0.6,  # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    ## from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19

    ## Always assume image [0,255]


    # Random Crop
    max_offest = 256 - const.frame_width # I already resized all images to 256
    rand = tf.random_uniform([2], minval=0, maxval=max_offest,dtype=tf.int32)
    height_offset = tf.cast(rand[0] , dtype=tf.int32)
    width_offest = tf.cast(rand[1] , dtype=tf.int32)
    images = tf.image.crop_to_bounding_box(images,height_offset, width_offest, const.frame_height , const.frame_width )

    # Color Augmentation
    # r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    # do_color_distortion = tf.less(r, color_aug_probability)
    # images = images / 255.0
    # images = tf.cond(do_color_distortion, lambda: tf.identity(
    #     apply_with_random_selector(images, lambda x, ordering: distort_color(images, ordering), num_cases=4)),
    #                 lambda: tf.identity(images))

    ## Adding Noise
    # noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
    # # images =
    # noisy_coin = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    # do_noise_img = tf.less(noisy_coin , noise_probability)
    # images = tf.cond(do_noise_img, lambda : images + noise,lambda: tf.identity(images))
    # images = tf.clip_by_value(images, 0.0, 1.0)
    # images = images * 255

    # resize = (const.frame_height, const.frame_width)
    # images = tf.image.resize_bilinear(images, resize)

    # My experiments showed that casting on GPU improves training performance
    if preprocess_func == 'inception_v1':
        print('Inception Format Augmentation')
        images = inception_preprocess(images)
    elif preprocess_func == 'densenet':
        print('DenseNet Format Augmentation')
        images = denseNet_preprocess(images)
    elif preprocess_func == 'vgg':
        print('VGG Format Augmentation')
        images = vgg_preprocess(images)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        # if vertical_flip:
        #     coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
        #     flip_transform = tf.convert_to_tensor(
        #         [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
        #     transforms.append(
        #         tf.where(coin,
        #                  tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
        #                  tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))
        #
        # if rotate > 0:
        #     angle_rad = rotate / 180 * math.pi
        #     angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
        #     transforms.append(
        #         tf.contrib.image.angles_to_projective_transforms(
        #             angles, height, width))
        #
        # if crop_probability > 0:
        #     crop_pct = tf.random_uniform([batch_size], crop_min_percent,
        #                                  crop_max_percent)
        #     left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
        #     top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
        #     crop_transform = tf.stack([
        #         crop_pct,
        #         tf.zeros([batch_size]), top,
        #         tf.zeros([batch_size]), crop_pct, left,
        #         tf.zeros([batch_size]),
        #         tf.zeros([batch_size])
        #     ], 1)
        #
        #     coin = tf.less(
        #         tf.random_uniform([batch_size], 0, 1.0), crop_probability)
        #     transforms.append(
        #         tf.where(coin, crop_transform,
        #                  tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    #resize = (const.frame_height, const.frame_width)
    #images = tf.image.resize_bilinear(images, resize)
    return images

# def vis_img(img,label,prefix,suffix):
#     imageio.imwrite(config.dump_path + prefix + '_' + str(label) + suffix + '.png',img)
#
#
# if __name__ == '__main__':
#     np.random.seed(10)
#     args = dict()
#     args['csv_file'] = config.train_csv_file
#     args['img_size'] = const.max_frame_size
#     args['gen_hot_vector'] = True
#
#
#     img_generator_class = locate(config.db_tuple_loader)
#     img_generator = img_generator_class(args)
#     imgs, lbls = img_generator.next();
#     print(np.min(imgs[0,:,:,:]),np.max(imgs[0,:,:,:]))
#     # quit()
#     tf_imgs = tf.constant(imgs)
#     from datetime import datetime
#     import time
#
#     np.random.seed((int)(time.time()))
#     tf.set_random_seed((int)(time.time()))
#     tf_result =  augment(imgs,resize=(const.frame_height, const.frame_width), horizontal_flip=False,vertical_flip=False,rotate=0,color_aug_probability=0,noise_probability=0.5)
#
#     sess = tf.InteractiveSession()
#     np_imgs = sess.run(tf_result)
#     print(np.min(np_imgs[0, :, :, :]), np.max(np_imgs[0, :, :, :]))
#     # print(rnd,height_offset)
#     # quit()
#     for i in range(np_imgs.shape[0]):
#         vis_img(imgs[i, :, :, :], i, 'p', 'o');
#         vis_img(np_imgs[i,:,:,:],i,'p','r1');