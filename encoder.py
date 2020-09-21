import tensorflow as tf
import numpy as np
from transformers import TFBertModel
from utils import*
from inceptionV3 import FeatureExtractor

class Encoder(tf.keras.layers.Layer):
    def __init__(self,feature_cache_path,img_path=None,pre_cal_batch_size=64):
        super(Encoder, self).__init__()       
        self.model = TFBertModel.from_pretrained('bert-base-uncased')
        self.features = FeatureExtractor(feature_cache_path,img_path,pre_cal_batch_size)
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(768,activation = None)
        ])
       


    def call(self,inputs):
        """
        batch_qst_tokens:[batch_size,max_len] int32
        batch_tr_img_ids:[batch_size] int32
        batch_te_img_ids:[batch_size] int32
        batch_tr_ans_tokens:[batch_size,num_ans] int32
        batch_te_ans_tokens:[batch_size,num_ans] int32
        """
        batch_qst_tokens,batch_tr_img_ids,batch_te_img_ids,batch_tr_ans_tokens,batch_te_ans_tokens = inputs
        qst_enc = self.model(batch_qst_tokens)#[batch_size,max_len,768] float32
        tr_img_enc = tf.concat([self.features[x] for x in batch_tr_img_ids],axis = 0)#[batch_size,8,8,2048] float32
        te_img_enc = tf.concat([self.features[x] for x in batch_te_img_ids],axis = 0)#[batch_size,8,8,2048] float32
        tr_img_enc = self.forward(tr_img_enc)#[batch_size,768]
        te_img_enc = self.forward(te_img_enc)#[batch_size,768]

        tr_ans_enc = tf.reduce_mean(self.model(batch_tr_ans_tokens),axis = 1)#[batch_size,768]
        te_ans_enc = tf.reduce_mean(self.model(batch_te_ans_tokens),axis = 1)#[batch_size,768]







if __name__ == "__main__":
    Encoder()