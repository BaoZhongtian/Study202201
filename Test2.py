import tqdm
import numpy
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MBart50Tokenizer,MBartForConditionalGeneration
from Loader_NCLS import NCLS_Loader

batch_size = 1

if __name__ == '__main__':
    train_data = NCLS_Loader(use_part='train', sample_number=100)
    # tokenizer = MT5Tokenizer.from_pretrained('C:/PythonProject/mt5-base')
    # model = MT5ForConditionalGeneration.from_pretrained(
    #     'E:/ProjectData/NCLS/MT5-finetuning-640Length-100K/00013998-Parameter')
    tokenizer = MBart50Tokenizer.from_pretrained('C:/PythonProject/mbart-large-50')
    model = MBartForConditionalGeneration.from_pretrained(
        'E:/ProjectData/NCLS/MonoLingual-MBART-finetuning-512Length-1K/00021998-Parameter')
    model.eval()

    for batch_index in tqdm.trange(0, len(train_data), batch_size):
        batch_article = [_['Article'] for _ in train_data[batch_index:batch_index + batch_size]]
        batch_summary = [_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]]
        print(batch_summary[0])

        batch_article = tokenizer.batch_encode_plus(
            batch_article, return_tensors='pt', max_length=1000, truncation=True, padding=True)
        batch_summary = tokenizer.batch_encode_plus(batch_summary)
        # print(numpy.shape(batch_article['input_ids']))
        # exit()
        print(len(batch_summary['input_ids'][0]))

        generated = model.generate(
            batch_article['input_ids'])
        print(generated)
        print(tokenizer.batch_decode(generated))
        exit()
    result = model.generate
