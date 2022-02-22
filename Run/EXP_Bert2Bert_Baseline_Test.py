import os
import numpy
import torch
import transformers
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 1

if __name__ == '__main__':
    model = transformers.EncoderDecoderModel.from_pretrained(
        'E:/ProjectData/NCLS/CrossLingual-EN2ZH-Summarization/Bert2Bert-OnlyDecoder-1K/00062998-Parameter')
    tokenizer = transformers.BertTokenizer.from_pretrained('C:/PythonProject/bert-base-multilingual-cased')
    model = model.cuda()

    train_data = NCLS_Loader(sample_number=100)

    model.eval()
    for batch_index in range(0, len(train_data), batch_size):
        model_inputs = tokenizer([_['Article'] for _ in train_data[batch_index:batch_index + batch_size]],
                                 max_length=512, padding="max_length", truncation=True, return_tensors='pt')

        # labels = tokenizer([_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]],
        #                    max_length=512, padding="max_length", truncation=True)['input_ids']

        labels = tokenizer([_['CrossLingualSummary'] for _ in train_data[batch_index:batch_index + batch_size]],
                           max_length=512, padding="max_length", truncation=True)['input_ids']
        for indexX in range(len(labels)):
            for indexY in range(len(labels[indexX])):
                if labels[indexX][indexY] == 0: labels[indexX][indexY] = -100

        result = model(input_ids=model_inputs['input_ids'].cuda(),
                       attention_mask=model_inputs['attention_mask'].cuda(), labels=torch.LongTensor(labels).cuda())
        print(result.loss)
        exit()

        result = model.generate(input_ids=model_inputs['input_ids'].cuda(),
                                attention_mask=model_inputs['attention_mask'].cuda(), early_stopping=False,
                                min_length=numpy.shape(labels)[1], max_length=numpy.shape(labels)[1], num_beams=4)
        print(tokenizer.batch_decode(result))
        print(result)
        exit()
