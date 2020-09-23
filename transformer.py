import tensorflow as tf
import numpy as np
from encoder import Encoder
from decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self,num_layers, d_model, num_heads, dff,
               maximum_position_encoding, rate=0.1):
        self.encoder = Encoder(r'D:\documents\coding\Data\coco\features',[r'D:\documents\coding\Data\coco\train2014',r'D:\documents\coding\Data\coco\val2014'])
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
               maximum_position_encoding, rate)
        


    def call(self,inputs):
        """
        batch_qst_tokens:[batch_size,max_len] int32
        batch_tr_img_ids:[batch_size] int32
        batch_te_img_ids:[batch_size] int32
        batch_tr_ans_tokens:[batch_size,num_ans] int32
        batch_te_ans_tokens:[batch_size,num_ans] int32
        batch_padding_mask:[batch_size,max_len] float32        
        batch_tr_ans_mask:[batch_size,num_ans,max_ans_len] float32
        batch_te_ans_mask:[batch_size,num_ans,max_ans_len] float32
        """
        encoder_output,test_ans_enc,attn_mask = self.encoder(inputs[:-2])
        decoder_output = self.decoder(encoder_output,attn_mask)
        predictions = decoder_output[:,3,:]
        return predictions,test_ans_enc

    def save(self):
        pass


    
