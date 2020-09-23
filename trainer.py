import tensorflow as tf
import numpy as np
from transformer import Transformer
from preprocess import DataPreprocesser
from vqa import VQA
from vqa_iter import VQAIter
from tokenizer import MyTokenizer
from encoder import Encoder
from decoder import Decoder
import tqdm
from loss import loss_cosine_similarity
from custom_schedule import get_cumtom_adam
import hyperparameters as hp
import pickle
class Trainer:
	def __init__(self,train_iter,model,batch_size,max_qst_len,max_ans_len,checkpoint_path):		
		self.train_iter = train_iter

		self.max_len = max(train_data.get_max_qst_len(),val_data.get_max_qst_len())
		self.preprocesser = DataPreprocesser(MyTokenizer(),self.max_len)
		self.model = model
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')
		self.optimizer = get_cumtom_adam()
		self.checkpoint_path  = checkpoint_path 
		self.ckpt = tf.train.Checkpoint(model=self.model,
							optimizer=self.optimizer)
		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)
		# if a checkpoint exists, restore the latest checkpoint.
		if self.ckpt_manager.latest_checkpoint:
			self.ckpt.restore(ckpt_manager.latest_checkpoint)
			print('Latest checkpoint restored!!')



	def train_step(self,batch):
		batch_qst_tokens,batch_tr_img_ids,batch_te_img_ids,batch_tr_ans_tokens,batch_te_ans_tokens = self.preprocesser(batch)
		
		#predictions:[batch_size,768]
		#test_ans_end:[batch_size,768]
		with tf.GradientTape() as tape:
			predictions,test_ans_enc = self.model(batch_qst_tokens,
								batch_tr_img_ids,
								batch_te_img_ids,
								batch_tr_ans_tokens,
								batch_te_ans_tokens)
			loss = loss_function(tar_real, predictions)
		gradients = tape.gradient(loss,self.model.trainable_variables)    
		optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
		self.train_loss(loss)
	

	def train(self,steps,steps_per_save,steps_per_chunk,steps_per_report):
		for s in range(steps):
			if s%steps_per_save == 0:		
				self.ckpt_manager.save()
				print('model saved')
			if s%steps == steps_per_chunk:
				self.train_iter.next_chunk()
			if s%steps_per_report == 0:
				print('Steps {} Loss {:.4f}'.format(s,train_loss.result()))
			self.train_step(self.train_iter.next())
		print('Steps {} Loss {:.4f}'.format(steps,train_loss.result()))
		self.model.save()
		print('model saved')
		print('training finished')
			

if __name__ == "__main__":
	#train_data = VQA(r'D:\documents\coding\Data\coco\v2_mscoco_train2014_annotations.json',
	#r'D:\documents\coding\Data\coco\v2_OpenEnded_mscoco_train2014_questions.json',
	#r'D:\documents\coding\Data\coco\train2014\COCO_train2014_{0}.jpg',
	#r'D:\documents\coding\Data\coco\v2_mscoco_train2014_complementary_pairs.json')
	train_data = VQA(r'D:\lgy\Document\Python\Data\coco\v2_mscoco_train2014_annotations.json',
	r'D:\lgy\Document\Python\Data\coco\v2_OpenEnded_mscoco_train2014_questions.json',
	r'D:\lgy\Document\Python\Data\coco\train2014\COCO_train2014_{0}.jpg')
	

	train_iter = VQAIter(train_data,train_data.getQuesIds(ansTypes = ['other','yes/no']),hp.batch_size,hp.num_chunks)
	
	max_qst_len = hp.max_qst_len
	max_ans_len = hp.max_ans_len

	model = Transformer(hp.num_layers,hp.d_model,hp.num_heads,hp.dff,max_qst_len+3,hp.dropout_rate)
	trainer = Trainer(train_iter,model,16,max_qst_len,max_ans_len)
	trainer.train(hp.steps,hp.steps_per_save,hp.steps_per_chunk,hp.steps_per_report)


