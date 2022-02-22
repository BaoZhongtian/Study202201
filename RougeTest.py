import os
import json
import numpy
from Loader_NCLS import NCLS_Loader
from rouge_score import rouge_scorer

if __name__ == '__main__':
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    test_data = NCLS_Loader(sample_number=100000, use_part='test')
    print(test_data[0]['Article'])
    test_data = [_['CrossLingualSummary'] for _ in test_data]

    load_path = 'E:/ProjectData/NCLS/EN2ZH-Summarization/MarianMT-All-Multi'
    epoch_score = []
    for epoch_name in os.listdir(load_path)[::-1]:
        if epoch_name.find('ResultBeam') == -1: continue
        total_predict = []
        for filename in os.listdir(os.path.join(load_path, epoch_name)):
            batch_predict = json.load(open(os.path.join(load_path, epoch_name, filename), 'r'))
            total_predict.extend(batch_predict)

        test_data = test_data[0:len(total_predict)]
        total_score = []
        for index in range(len(total_predict)):
            sample_score = scorer.score(prediction=total_predict[index], target=test_data[index])
            total_score.append(
                [sample_score['rouge1'].fmeasure, sample_score['rouge2'].fmeasure, sample_score['rougeL'].fmeasure])
        print(numpy.average(total_score, axis=0))
        epoch_score.append(numpy.average(total_score, axis=0))

        for index in range(5):
            print(test_data[index])
            print(total_predict[index])
            print()
        exit()
    # print(total_predict)
    print(numpy.max(numpy.array(epoch_score), axis=0))
