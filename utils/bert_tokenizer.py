import tiktoken
import torch
import transformers as ppb

class BERTTokenizer(object):

    def __init__(self):
        super(BERTTokenizer, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    
    def __call__(self,text:str=None) -> torch.LongTensor:
        tokens = self.tokenizer.encode(text)
        return tokens
       
    
