import tensorflow as tf
import numpy as np

class Transformer(tf.keras.Model):
    def __init__(self,encoder,decoder):
        self.encoder = encoder
        self.decoder = decoder
        

    def call(self,inputs):
        """
        batch_qst_tokens:[batch_size,max_len] int32
        batch_tr_img_ids:[batch_size] int32
        batch_te_img_ids:[batch_size] int32
        batch_tr_ans_tokens:[batch_size,num_ans] int32
        batch_te_ans_tokens:[batch_size,num_ans] int32
        batch_padding_mask:[batch_size,max_len] float32        
        """
        encoder_output,test_ans_enc,attn_mask = self.encoder(inputs)
        decoder_output = self.decoder(encoder_output,attn_mask)
        predictions = decoder_output[:,3,:]
        return predictions,test_ans_enc

    def save(self):
        pass


    
