import tensorflow as tf
import numpy as np
from transformers import*
from utils import*

class MyTokenizer:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.TRICS = '[unused0]'
        self.TEICS = '[unused1]'
        self.TRICI = 1
        self.TEICI = 2
        
    def encode(self,questions,train_labels):
        """
        questions:[string]
        trains:[[string]]

        return -> dict{input_ids,token_type_ids,attention_mask,img_ids}
        """
        txts = []
        
        for i in range(len(questions)):
            txt = [questions[i]]
            for label in train_labels:
                txt.append(self.TRIC)
                txt.append(label)
                txt.append("[SEP]")
            txt.append(self.TEIC)
            txt.append("[SEP]")
            txts.append(''.join(txt))
        
        batched = self.bert_tokenizer(txts, padding=True, truncation=True)
        img_ids = [[i for i,x in enumerate(y) if x == self.TEICI or x == self.TRICI] for y in batched['input_ids']]
        batched['img_ids'] = img_ids
        return batched








if __name__ == "__main__":
    pass


