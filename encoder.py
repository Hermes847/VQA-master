import tensorflow as tf
import numpy as np
from transformers import*
from utils import*


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.model = TFBertModel.from_pretrained('bert-base-uncased')
    def call(self,inputs):
        return self.model(inputs)



if __name__ == "__main__":
    Encoder()