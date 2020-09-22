import tensorflow as tf
import numpy as np

def loss_cosine_similarity(real, pred):
  return tf.reduce_mean(tf.losses.cosine_similarity(real,pred))


if __name__ =="__main__":
    pass



