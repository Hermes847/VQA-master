__author__ = 'aagrawal'
__version__ = '0.9'

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy
import sys
import re
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import progressbar as pbar
class VQA:
	def __init__(self, annotation_file=None, question_file=None,img_path_format = None,complementary_pairs = None):
		"""
       	Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
		"""
        # load dataset
		self.img_path_format = img_path_format
		self.dataset = {}
		self.questions = {}
		self.qa = {}
		self.qqa = {}
		self.imgToQA = {}
		self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
		self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}
		self.articles     = ['a',
								'an',
								'the'
							]

		self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
		self.commaStrip   = re.compile("(\d)(\,)(\d)")
		self.punct        = [';', r"/", '[', ']', '"', '{', '}',
								'(', ')', '=', '+', '\\', '_', '-',
								'>', '<', '@', '`', ',', '?', '!']
		self.pos_to_neg = {}
		self.pairs = {}
		if not annotation_file == None and not question_file == None:
			print('loading VQA annotations and questions into memory...')
			time_t = datetime.datetime.utcnow()
			dataset = json.load(open(annotation_file, 'r'))
			questions = json.load(open(question_file, 'r'))
			print(datetime.datetime.utcnow() - time_t)
			self.dataset = dataset
			self.questions = questions
			self.createIndex()
		if complementary_pairs:
			self.create_complementary_pairs(complementary_pairs)
			
	def create_complementary_pairs(self):
		pos_to_neg = {}
		for q in self.qa:
			if self.qa[q]['score']>0.5:
				pos_id = q
				neg_id = 900000000 + pos_id
			else:
				neg_id = q
				pos_id = pos_id - 900000000
			if pos_id in self.qqa and neg_id in self.qqa and pos_id not in pos_to_neg:
				if self.qqa[pos_id]['question'] == self.qqa[neg_id]['question']:
					pos_to_neg[pos_id] = neg_id
		self.pos_to_neg = pos_to_neg

	def create_complementary_pairs(self,cp_path):
		cps = json.load(open(cp_path,'r'))
		for a,b in cps:
			self.pairs[a] = b
			self.pairs[b] = a
			
	def processPunctuation(self, inText):
		outText = inText
		for p in self.punct:
			if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
				outText = outText.replace(p, '')
			else:
				outText = outText.replace(p, ' ')	
		outText = self.periodStrip.sub("",
									  outText,
									  re.UNICODE)
		return outText
	
	def processDigitArticle(self, inText):
		outText = []
		tempText = inText.lower().split()
		for word in tempText:
			word = self.manualMap.setdefault(word, word)
			if word not in self.articles:
				outText.append(word)
			else:
				pass
		for wordId, word in enumerate(outText):
			if word in self.contractions: 
				outText[wordId] = self.contractions[word]
		outText = ' '.join(outText)
		return outText

	def create_vocab(self,ques_ids = None,answer_only = False,show_progress = True):
		vocab = {}
		def func(question_id):
			if not answer_only:
				txt = self.qa[question_id]['question']
				txt = self.processPunctuation(txt)
				txt = self.processDigitArticle(txt)
				for s in txt.split(' '):
					if s in vocab:
						vocab[s]+=1
					else:
						vocab[s] = 1
			for a in self.qa[question_id]['answers']:
				txt = a['answer']
				txt = self.processPunctuation(txt)
				txt = self.processDigitArticle(txt)
				for s in txt.split(' '):
					if s in vocab:
						vocab[s]+=1
					else:
						vocab[s] = 1
		ques = ques_ids if ques_ids else self.qa
		if show_progress:
			for question_id in pbar.ProgressBar(len(self.qa),['create vocab...',pbar.Percentage(),pbar.Bar(),pbar.ETA()])(ques):
				func(question_id)
		else:
			for question_id in ques:
				func(question_id)
		return vocab#sorted(vocab,key = lambda x:vocab[x],reverse = True)	
	
	def get_ques_types(self):
		types = set()
		for _,x in self.qa.items():
		    types.add(x['question_type'])
		return list(types)

	def get_answer_types(self):
		types = set()
		for _,x in self.qa.items():
		    types.add(x['answer_type'])
		return list(types)

	def createIndex(self):
        # create index
		print('creating index...')
		imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
		qa =  {ann['question_id']:       [] for ann in self.dataset['annotations']}
		qqa = {ann['question_id']:       [] for ann in self.dataset['annotations']}
		for ann in self.dataset['annotations']:
			imgToQA[ann['image_id']] += [ann]
			qa[ann['question_id']] = ann
			#yes_count = 0
			#no_count = 0
			#for x in ann['answers']:
			#	if x['answer'] == 'yes':
			#		yes_count+=1
			#	elif x['answer'] == 'no':
			#		no_count+=1
			#qa[ann['question_id']]['score'] = yes_count/(yes_count+no_count)
		for qst in self.questions['questions']:
			qa[qst['question_id']].update(qst)
			#qqa[ques['question_id']] = ques
			
		print('index created!')

 		# create class members
		self.qa = qa
		self.qqa = qqa
		self.imgToQA = imgToQA

	def info(self):
		"""
		Print information about the VQA annotation file.
		:return:
		"""
		for key, value in self.datset['info'].items():
			print('%s: %s'%(key, value))


	def getQuesIds(self,quesTypes=[], ansTypes=[]):
		ids = []
		for index in self.qa:
		    if ((not quesTypes) or self.qa[index]['question_type'] in quesTypes)\
			and ((not ansTypes) or self.qa[index]['answer_type'] in ansTypes):
		        ids.append(index)
		return ids


	#def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
	#	"""
	#	Get question ids that satisfy given filter conditions. default skips that filter
	#	:param 	imgIds    (int array)   : get question ids for given imgs
	#			quesTypes (str array)   : get question ids for given question types
	#			ansTypes  (str array)   : get question ids for given answer types
	#	:return:    ids   (int array)   : integer array of question ids
	#	"""
	#	imgIds 	  = imgIds    if type(imgIds)    == list else [imgIds]
	#	quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
	#	ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

	#	if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
	#		anns = self.dataset['annotations']
	#	else:
	#		if not len(imgIds) == 0:
	#			anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],[])
	#		else:
	#			anns = self.dataset['annotations']
	#		anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
	#		anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
	#	ids = [ann['question_id'] for ann in anns]
	#	return ids

	def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
		"""
		Get image ids that satisfy given filter conditions. default skips that filter
		:param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
		:return: ids     (int array)   : integer array of image ids
		"""
		quesIds   = quesIds   if type(quesIds)   == list else [quesIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

		if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(quesIds) == 0:
				anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa],[])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['image_id'] for ann in anns]
		return ids

	def loadQA(self, ids=[]):
		"""
		Load questions and answers with the specified question ids.
		:param ids (int array)       : integer ids specifying question ids
		:return: qa (object array)   : loaded qa objects
		"""
		if type(ids) == list:
			return [self.qa[id] for id in ids]
		elif type(ids) == int:
			return [self.qa[ids]]

	def showQA(self, anns):
		"""
		Display the specified annotations.
		:param anns (array of object): annotations to display
		:return: None
		"""
		if len(anns) == 0:
			return 0
		for ann in anns:
			quesId = ann['question_id']
			#print("Question: %s" %(self.qqa[quesId]['question']))
			print("Question: %s" %(ann['question']))
			for ans in ann['answers']:
				print("Answer %d: %s" %(ans['answer_id'], ans['answer']))
		
	def loadRes(self, resFile, quesFile):
		"""
		Load result file and return a result object.
		:param   resFile (str)     : file name of result file
		:return: res (obj)         : result api object
		"""
		res = VQA()
		res.questions = json.load(open(quesFile))
		res.dataset['info'] = copy.deepcopy(self.questions['info'])
		res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
		res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
		res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
		res.dataset['license'] = copy.deepcopy(self.questions['license'])

		print('Loading and preparing results...     ')
		time_t = datetime.datetime.utcnow()
		anns    = json.load(open(resFile))
		assert type(anns) == list, 'results is not an array of objects'
		annsQuesIds = [ann['question_id'] for ann in anns]
		assert set(annsQuesIds) == set(self.getQuesIds()), \
		'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
		for ann in anns:
			quesId 			     = ann['question_id']
			if res.dataset['task_type'] == 'Multiple Choice':
				assert ann['answer'] in self.qqa[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
			qaAnn                = self.qa[quesId]
			ann['image_id']      = qaAnn['image_id'] 
			ann['question_type'] = qaAnn['question_type']
			ann['answer_type']   = qaAnn['answer_type']
		print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

		res.dataset['annotations'] = anns
		res.createIndex()
		return res

	def plot_img(self,img_id):
		imgFilename = self.img_path_format.format(str(img_id).zfill(12))
		if os.path.isfile(imgFilename):
			I = io.imread(imgFilename)
			plt.imshow(I)
			plt.axis('off')
			plt.show()

	def take_a_look(self,count,quesIds=[], quesTypes=[], ansTypes=[],show_type = 'random'):
		"""
			show_type:
			random 
			pair 
			same_qst
		"""
		annIds = self.getQuesIds(quesIds,quesTypes,ansTypes);   
		anns = self.loadQA(annIds)
		for _ in range(count):
			randomAnn = random.choice(anns)
			self.showQA([randomAnn])
			self.plot_img(randomAnn['image_id'])
			if show_type == 'pair':
				pair = self.qa[self.pairs[randomAnn['question_id']]]
				self.showQA([pair])
				self.plot_img(pair['image_id'])
			elif show_type == 'same_qst':
				qst = randomAnn['question']
				for pair in [y for x,y in self.qa.items() if y['question'] == qst]:
					if pair['question_id']!=randomAnn['question_id']:
						self.showQA([pair])
						self.plot_img(pair['image_id'])

	def get_img_paths(self,ids):
		img_pahts = []
		for index in ids:
			path = self.img_path_format.format(str(self.qa[index]['image_id']).zfill(12))
			img_pahts.append(path)	
		return img_pahts

	def get_max_qst_len(self):
		return max(self.qa,key = lambda x:len(x['question'].split()))



if __name__ == "__main__":
	ds1 = VQA(r'D:\documents\coding\Data\coco\v2_mscoco_train2014_annotations.json',
		   r'D:\documents\coding\Data\coco\v2_OpenEnded_mscoco_train2014_questions.json',
		   r'D:\documents\coding\Data\coco\train2014\COCO_train2014_{0}.jpg',
		   r'D:\documents\coding\Data\coco\v2_mscoco_train2014_complementary_pairs.json')
	ds2 = VQA(r'D:\documents\coding\Data\coco\v2_mscoco_val2014_annotations.json',
		   r'D:\documents\coding\Data\coco\v2_OpenEnded_mscoco_val2014_questions.json',
		   r'D:\documents\coding\Data\coco\train2014\COCO_val2014_{0}.jpg',
		   r'D:\documents\coding\Data\coco\v2_mscoco_val2014_complementary_pairs.json')
	#vqa.take_a_look(100,show_type = 'same_qst')
	ids2 = ds2.getQuesIds(ansTypes = ['other','yes/no'])
	vocab_val = ds2.create_vocab(ids,True,False)
	pass
