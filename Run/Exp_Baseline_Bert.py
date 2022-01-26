import os
import numpy
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    