

class DataPreprocesser:
    def __init__(self,tokenizer,max_qst_len):
        self.tokenizer = tokenizer
        self.max_qst_len=max_qst_len
     
   
    def __call__(self,batch):
        batch_qst_tokens = []
        batch_tr_img_ids = []
        batch_te_img_ids = []
        batch_tr_ans_tokens = []
        batch_te_ans_tokens = []
        for qst,(tr_img_id,tr_ans),(te_img_id,te_ans) in batch:
            batch_qst_tokens.append(self.tokenizer(qst,self.max_qst_len))
            batch_tr_img_ids.append(tr_img_id)
            batch_te_img_ids.append(te_img_id)
            tr_ans_tokens = []
            for x in tr_ans:
                tr_ans_tokens.append(self.tokenizer(x))
            batch_tr_ans_tokens.append(tr_ans_tokens)
            te_ans_tokens = []
            for x in te_ans:
                te_ans_tokens.append(self.tokenizer(x))
            batch_te_ans_tokens.append(batch_te_ans_tokens)
        return np.array(batch_qst_tokens,dtype = 'int32'),\
            np.array(batch_tr_img_ids,dtype = 'int32'),\
            np.array(batch_te_img_ids,dtype = 'int32'),\
            np.array(batch_tr_ans_tokens,dtype = 'int32'),\
            np.array(batch_te_ans_tokens,dtype = 'int32')


            





if __name__ == "__main__":
    pass

