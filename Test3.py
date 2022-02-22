import os
import numpy
import torch
import json
import tqdm
import transformers
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader

batch_size = 1

if __name__ == '__main__':
    tokenizer = transformers.MarianTokenizer.from_pretrained("C:/PythonProject/opus-mt-en-zh")

    load_path = 'E:/ProjectData/NCLS/EN2ZH-Summarization/MarianMT-10K-Multi/00179992-Parameter'
    model = transformers.MarianMTModel.from_pretrained(load_path)
    model.eval()
    model = model.cuda()

    test_data = NCLS_Loader(sample_number=10, use_part='train')

    for batch_index in tqdm.trange(0, len(test_data), batch_size):
        model_inputs = tokenizer([_['Article'] for _ in test_data[batch_index:batch_index + batch_size]],
                                 max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        labels = [_['CrossLingualSummary'] for _ in test_data[batch_index:batch_index + batch_size]]

        outputs = model.generate(input_ids=model_inputs['input_ids'].cuda(),
                                 attention_mask=model_inputs['attention_mask'].cuda())
        predict_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(labels)
        print(predict_sentence)
        print('\n')
