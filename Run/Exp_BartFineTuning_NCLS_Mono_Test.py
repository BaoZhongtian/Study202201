import os
import numpy
import json
import tqdm
import torch
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader
from transformers import BartForConditionalGeneration, BartTokenizer, EncoderDecoderModel, BertTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 2

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('C:/PythonProject/bert-base-uncased')

    train_data = NCLS_Loader(use_part='train', sample_number=100)
    parameter_path = 'E:/ProjectData/NCLS/bert2bert-finetuning-512Length-100K/'

    # for foldname in os.listdir(parameter_path):
    #     if foldname.find('TestResult') != -1: continue
    #     if foldname.find('Parameter') == -1: continue

    parameter_name = '00145186-Parameter'
    if not os.path.exists(os.path.join(parameter_path, parameter_name + '-TestResult')):
        os.makedirs(os.path.join(parameter_path, parameter_name + '-TestResult'))
    # model = BartForConditionalGeneration.from_pretrained(parameter_path + parameter_name)
    model = EncoderDecoderModel.from_pretrained(parameter_path + parameter_name)
    model.decoder_start_token_id = 101
    model.config.pad_token_id = 0
    model.eval()
    model = model.cuda()

    for batch_index in tqdm.trange(0, len(train_data), batch_size):
        # if os.path.exists(
        #         os.path.join(parameter_path, parameter_name + '-TestResult',
        #                      '%08d-640.json' % batch_index)): continue
        # with open(os.path.join(parameter_path, parameter_name + '-TestResult', '%08d-640.json' % batch_index), 'w'):
        #     pass
        batch_article = [_['Article'] for _ in train_data[batch_index:batch_index + batch_size]]
        batch_summary = [_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]]
        batch_article = tokenizer.batch_encode_plus(
            batch_article, return_tensors='pt', max_length=512, truncation=True, padding=True)
        batch_summary = tokenizer.batch_encode_plus(
            batch_summary, return_tensors='pt', truncation=True, padding=True)

        # result = model(
        #     input_ids=batch_article['input_ids'].cuda(), attention_mask=batch_article['attention_mask'].cuda(),
        #     labels=batch_summary['input_ids'].cuda(), return_dict=True)
        result = model.generate(batch_article['input_ids'].cuda(), bos_token_id=101, eos_token_id=102,
                                repetition_penalty=1.0).detach().cpu().numpy().tolist()
        print(result)
        exit()

        predict = torch.argmax(result['logits'], dim=-1)
        print(tokenizer.batch_decode(predict))
        exit()

        # article_ids, article_label = batch_article['input_ids'], batch_article['mlm_label']
        # result = model.generate(batch_article['input_ids'].cuda(), bos_token_id=101, eos_token_id=102).detach().cpu().numpy().tolist()
        # for sample in tokenizer.batch_decode(result):
        #     print(sample)
        exit()
        json.dump(result,
                  open(os.path.join(parameter_path, parameter_name + '-TestResult', '%08d-640.json' % batch_index),
                       'w'))
