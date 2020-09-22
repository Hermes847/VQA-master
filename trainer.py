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


class Trainer:
	def __init__(self,train_iter,val_iter,model,batch_size):		
		self.train_iter = train_iter
		self.val_iter = val_iter

		self.max_len = max(train_data.get_max_qst_len(),val_data.get_max_qst_len())
		self.preprocesser = DataPreprocesser(MyTokenizer(),self.max_len)
		self.model = model
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')


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
	


	def train(self,step,step_per_save,step_per_chunk,step_per_report):
		for s in range(step):
			if s%step_per_save == 0:		
				self.model.save()
				print('model saved')
			if s%step == step_per_chunk:
				self.train_iter.next_chunk()
			if s%step_per_report == 0:
				print('Steps {} Loss {:.4f}'.format(s,train_loss.result()))
			self.train_step(self.train_iter.next())
		print('Steps {} Loss {:.4f}'.format(s,train_loss.result()))
		self.model.save()
		print('model saved')
		print('training finished')
			
	def test(self):
		pass

if __name__ == "__main__":
	train_data = VQA(r'D:\documents\coding\Data\coco\v2_mscoco_train2014_annotations.json',
	r'D:\documents\coding\Data\coco\v2_OpenEnded_mscoco_train2014_questions.json',
	r'D:\documents\coding\Data\coco\train2014\COCO_train2014_{0}.jpg',
	r'D:\documents\coding\Data\coco\v2_mscoco_train2014_complementary_pairs.json')
	val_data = VQA(r'D:\documents\coding\Data\coco\v2_mscoco_val2014_annotations.json',
	r'D:\documents\coding\Data\coco\v2_OpenEnded_mscoco_val2014_questions.json',
	r'D:\documents\coding\Data\coco\train2014\COCO_val2014_{0}.jpg',
	r'D:\documents\coding\Data\coco\v2_mscoco_val2014_complementary_pairs.json')
	batch_size = 16
	train_iter = VQAIter(train_data,train_data.getQuesIds(ansTypes = ['other','yes/no']),batch_size)
	val_iter = VQAIter(val_data,val_data.getQuesIds(ansTypes = ['other','yes/no']),batch_size)
	encoder = Encoder(r'D:\documents\coding\Data\coco\features',[r'D:\documents\coding\Data\coco\train2014',r'D:\documents\coding\Data\coco\val2014'])
	decoder = Decoder()
	model = Transformer(encoder,decoder)
	trainer = Trainer(train_iter,val_iter,model)
	trainer.train(1000000,1000,10000,10)


