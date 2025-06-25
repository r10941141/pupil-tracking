import tensorflow as tf

def adjust_brightness(image, delta_range=(-0.2, 0.2)):
    delta = tf.random.uniform([], *delta_range)
    return tf.image.adjust_brightness(image, delta)

def adjust_contrast(image, lower=0.8, upper=1.2):
    factor = tf.random.uniform([], lower, upper)
    return tf.image.adjust_contrast(image, factor)
