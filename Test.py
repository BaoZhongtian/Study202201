import os
import json
import tqdm
import numpy
from rouge_score import rouge_scorer
from Loader_NCLS import NCLS_Loader
from transformers import BartTokenizer

if __name__ == '__main__':
    test_data = NCLS_Loader(use_part='test', sample_number=99999)

    tokenizer = BartTokenizer.from_pretrained('C:/ProjectData/bart-large-cnn')
    load_path = 'E:/ProjectData/NCLS/bart-finetuning-640Length-100K/00326988-Parameter-TestResult'
    total_data = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    for filename in tqdm.tqdm(os.listdir(load_path)):
        current_data = json.load(open(os.path.join(load_path, filename)))
        total_data.extend(tokenizer.batch_decode(current_data, skip_special_tokens=True))
    total_score = []
    for index in range(len(total_data)):
        result = scorer.score(target=test_data[index]['Summary'], prediction=total_data[index])
        total_score.append([result['rouge1'].fmeasure, result['rouge2'].fmeasure, result['rougeL'].fmeasure])
    print(numpy.average(total_score, axis=0))
