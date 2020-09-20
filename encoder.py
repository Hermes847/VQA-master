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
        self.forward = tf.keras.layers.Dense(768,activation = None)



    def call(self,inputs):
        batch_qst_tokens,batch_tr_img_ids,batch_te_img_ids,batch_tr_ans_tokens,batch_te_ans_tokens = inputs
        qst_enc = self.model(batch_qst_tokens)
        tr_img_enc = [self.features[x] for x in batch_tr_img_ids]
        te_img_enc = [self.features[x] for x in batch_te_img_ids]
        tr_ans_enc = tf.reduce_mean(self.model(batch_tr_ans_tokens),axis = 0)
        te_ans_enc = tf.reduce_mean(self.model(batch_te_ans_tokens),axis = 0)






if __name__ == "__main__":
    Encoder()