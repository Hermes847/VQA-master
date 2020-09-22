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
        self.forward = tf.keras.Sequential([tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(760,activation = None)])
       
       
    def call(self,inputs):
        """
        batch_qst_tokens:[batch_size,max_len] int32
        batch_tr_img_ids:[batch_size] int32
        batch_te_img_ids:[batch_size] int32
        batch_tr_ans_tokens:[batch_size,num_ans] int32
        batch_te_ans_tokens:[batch_size,num_ans] int32
        batch_padding_mask:[batch_size,max_len] float32  
        """
        batch_qst_tokens,batch_tr_img_ids,batch_te_img_ids,batch_tr_ans_tokens,batch_te_ans_tokens = inputs
        qst_enc = self.model(batch_qst_tokens)#[batch_size,max_len,768] float32
        tr_img_enc = tf.concat([self.features[x] for x in batch_tr_img_ids],axis = 0)#[batch_size,8,8,2048] float32
        te_img_enc = tf.concat([self.features[x] for x in batch_te_img_ids],axis = 0)#[batch_size,8,8,2048] float32
        tr_img_enc = self.forward(tr_img_enc)#[batch_size,760]
        te_img_enc = self.forward(te_img_enc)#[batch_size,760]
        tr_img_enc = tf.concat(tf.one_hot([0] * batch_size,8),tr_img_enc,axis = -1)#[batch_size,768]
        te_img_enc = tf.concat(tf.one_hot([1] * batch_size,8),te_img_enc,axis = -1)#[batch_size,768]
        tr_ans_enc = tf.reduce_mean(self.model(batch_tr_ans_tokens),axis = 1)#[batch_size,768]
        te_ans_enc = tf.reduce_mean(self.model(batch_te_ans_tokens),axis = 1)#[batch_size,768]
        tr_img_enc = tf.expand_dims(tr_img_enc,axis = 1)#[batch_size,1,768]
        te_img_enc = tf.expand_dims(te_img_enc,axis = 1)#[batch_size,1,768]
        tr_ans_enc = tf.expand_dims(tr_ans_enc,axis = 1)#[batch_size,1,768]
               
        tr_seq = tf.concat([tr_img_enc,tr_ans_enc,te_img_enc,qst_enc],axis = 1)#[batch_size,max_len+3,768]
        batch_padding_mask = tf.concat([tf.ones([batch_size,3]),batch_padding_mask],axis = -1)

        #[train_img,train_ans,test_img,cls,qst_1,qst_2,...,qst_n,sep,pad,...,pad]
        return tr_seq,te_ans_enc,batch_padding_mask







if __name__ == "__main__":
    Encoder()