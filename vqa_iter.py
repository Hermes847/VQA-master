
import vqa
import utils as utl 
import random
class VQAIter:
	def __init__(self,ds,ids,batch_size,num_chunks):
		self.ds = ds
		self.batch_size = batch_size
		self.ids = ids
		self.groups = [x[1] for x in utl.group_by(self.ids,key = lambda x:self.ds.qa[x]['question'],value = lambda x:x).items()]
		self.cur_chunk = 0
		self.num_chunks=num_chunks
		self.chunk_size = len(groups)//num_chunks
		self.index = 0

	def get_cur_chunk(self):
		return groups[self.cur_chunk*self.chunk_size:self.cur_chunk*(self.chunk_size+1)]

	def next_chunk(self):		
		self.cur_chunk = (self.cur_chunk+1)%self.num_chunks
		self.index = self.cur_chunk*self.chunk_size

	def shuffle_cur_chunk(self):
		utl.shuffle_segment(self.groups,self.index,self.chunk_size)
		

	def next(self):
		return self.__next__()

	def __iter__(self):
		return self

	def __next__(self):
		batch = []
		for _ in range(self.batch_size):
			qsts = self.groups[self.index]
			train_index = random.randint(0,len(qsts)//2-1)
			test_index = random.randint(len(qsts)//2,len(qsts)-1)
			batch.append(
				self.ds.qa[test_index]['question'],
				(					
					seld.ds.qa[train_index]['image_id'],
					[x['answer'] for x in self.ds.qa[train_index]['answers']]			
				),
				(					
					self.ds.qa[test_index]['image_id'],
					[x['answer'] for x in self.ds.qa[test_index]['answers']]			
				)
			)			
			self.index+=1
			if self.index >= (self.cur_chunk+1)*self.chunk_size:
				self.index = self.cur_chunk*self.chunk_size
				self.shuffle_cur_chunk()
		return batch

	