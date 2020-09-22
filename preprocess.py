

class DataPreprocesser:
    def __init__(self,tokenizer,max_qst_len,max_ans_len):
        self.tokenizer = tokenizer
        self.max_qst_len=max_qst_len
     
   
    def __call__(self,batch):
        batch_qst_tokens = []
        batch_qst_mask = []
        batch_tr_img_ids = []
        batch_te_img_ids = []
        batch_tr_ans_tokens = []
        batch_te_ans_tokens = []
        batch_tr_ans_mask = []
        batch_te_ans_mask = []


        for qst,(tr_img_id,tr_ans),(te_img_id,te_ans) in batch:
            tokens = self.tokenizer(qst,self.max_qst_len)
            batch_qst_tokens.append(tokens['input_ids'])
            batch_qst_mask.append(tokens['attention_mask'])
            batch_tr_img_ids.append(tr_img_id)
            batch_te_img_ids.append(te_img_id)
            tr_ans_tokens = []
            tr_ans_mask = []

            for x in tr_ans:
                tokens = self.tokenizer(x,max_ans_len,False,False)
                tr_ans_tokens.append(tokens['input_ids'])
                tr_ans_mask.append(tokens['attention_mask'])
            batch_tr_ans_tokens.append(tr_ans_tokens)
            batch_tr_ans_mask.append(tr_ans_mask)
            te_ans_tokens = []
            te_ans_mask = []
            for x in te_ans:
                tokens = self.tokenizer(x)
                te_ans_tokens.append(tokens['input_ids'])
                te_ans_mask.append(tokens['attention_mask'])
            batch_te_ans_tokens.append(batch_te_ans_tokens)
            batch_te_ans_mask.append(te_ans_mask)

        return np.array(batch_qst_tokens,dtype = 'int32'),\
            np.array(batch_tr_img_ids,dtype = 'int32'),\
            np.array(batch_te_img_ids,dtype = 'int32'),\
            np.array(batch_tr_ans_tokens,dtype = 'int32'),\
            np.array(batch_te_ans_tokens,dtype = 'int32'),\
            np.array(batch_qst_mask,dtype = 'float32'),\
            np.array(batch_tr_ans_mask,dtype = 'float32'),\
            np.array(batch_te_ans_mask,dtype = 'float32')



            





if __name__ == "__main__":
    pass

