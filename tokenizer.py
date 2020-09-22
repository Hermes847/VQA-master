import tensorflow as tf
import numpy as np
from transformers import*
from utils import*

class MyTokenizer:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self,x,max_length = None,add_cls = True,add_sep = True):
        """
        questions:string

        return -> dict{input_ids,token_type_ids,attention_mask,img_ids}
        """

        tokens = self.bert_tokenizer(x, padding='max_length', max_length=max_length)
        if not add_cls:
            for _,x in tokens.items():
                x = x[1:]
        if not add_sep:
            for _,x in tokens.items():
                x = x[:-1]
        return tokens


if __name__ == "__main__":
    pass


