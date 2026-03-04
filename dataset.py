from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


print(tf.__version__)
def load_image(path):

    tf_read = tf.io.read_file(path)
    tf_img_decode = tf.image.decode_image(tf_read,channels = 3, expand_animations = False)
    tf_img_float = tf.image.convert_image_dtype(tf_img_decode, tf.float32)
    tf_img_float.set_shape([None,None,3])

    return tf_img_float
    

def make_hr_patch(img,hr_size):
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    condition = tf.logical_or(h < hr_size, w< hr_size)
    
    def resize_img():
        new_h = tf.maximum(h,hr_size)
        new_w = tf.maximum(w,hr_size)
        
        new_img = tf.image.resize(img, size =[new_h,new_w], method = 'bilinear')
        return new_img
        
    def no_resize():
        return img

    img = tf.cond(condition,resize_img,no_resize)

    hr_patch = tf.image.random_crop(img, size=[hr_size,hr_size,3])
    return hr_patch

def make_lr_patch(hr, scale):
    h = tf.shape(hr)[0]
    w = tf.shape(hr)[1]

    new_h = h // scale
    new_w = w// scale

    lr_small = tf.image.resize(hr,size = [new_h,new_w], method = 'bicubic')
    lr_small = tf.clip_by_value(lr_small, 0.0, 1.0)

    return lr_small

def make_pair(path,hr_size,scale):
    img = load_image(path)
    hr = make_hr_patch(img, hr_size)
    lr = make_lr_patch(hr,scale)

    return lr,hr

def make_sr_dataset(paths,hr_size,scale,batchs_size,training):
    shuffle_buffer = 1000

    ds = tf.data.Dataset.from_tensor_slices(paths)
    
    if training:
        ds = ds.shuffle(shuffle_buffer,reshuffle_each_iteration = True)

    ds= ds.map(lambda p: make_pair(p,hr_size,scale),num_parallel_calls = tf.data.AUTOTUNE)

    if training:
        ds = ds.batch(batchs_size,drop_remainder = True)
    else:
        ds = ds.batch(batchs_size,drop_remainder = False)

    ds = ds.prefetch(tf.data.AUTOTUNE)

        
    return ds
    
