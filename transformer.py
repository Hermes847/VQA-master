import tensorflow as tf
import numpy as np
from utils import*
from encoder import Encoder
from tokenizer import MyTokenizer

class Transformer(tf.keras.Model):
    def __init__(self,encoder,decoder):
        self.encoder = encoder
        self.decoder = decoder


    def call(self,inputs):
        pass

    def save(self):
        pass


    
