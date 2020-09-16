import tensorflow as tf
import numpy as np
from utils import*
from encoder import Encoder
from tokenizer import MyTokenizer

class Transformer(tf.keras.Model):
    def __init__(self):
        self.tokenizer = MyTokenizer()
        self.encoder = Encoder()
    def call(self,inputs):
        pass

    
