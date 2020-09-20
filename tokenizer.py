import tensorflow as tf
import numpy as np
from transformers import*
from utils import*

class MyTokenizer:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self,x,max_length = None):
        """
        questions:string

        return -> dict{input_ids,token_type_ids,attention_mask,img_ids}
        """
        if self.question.find(' ') == -1:          
            if x == '[IMG]':
                return 1
            else:
                return self.bert_tokenizer(x)[1]
        else:
            return self.bert_tokenizer(x, padding='max_length', max_length=max_length)

if __name__ == "__main__":
    pass


