import os
import json
import tqdm
import pickle
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

top_n = 100

if __name__ == '__main__':
    vectorizer = pickle.load(open('TFIDF_CountVectorizer.pkl', 'rb'))
    transformer = pickle.load(open('TFIDF_TfidfTransformer.pkl', 'rb'))
    reverse_vocabulary = {}
    for key in vectorizer.vocabulary_:
        reverse_vocabulary[vectorizer.vocabulary_[key]] = key

    load_path = 'D:/ProjectData/CNNDM/Step0_TextTruncation/'
    save_path = 'D:/ProjectData/CNNDM/Step2_KeywordsGeneration/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    for batch_start in tqdm.trange(0, len(os.listdir(load_path)), 100):
        if os.path.exists(save_path + 'CNNDM_SalientWords_Train_%d_Top100.json' % batch_start): continue
        with open(save_path + 'CNNDM_SalientWords_Train_%d_Top100.json' % batch_start, 'w'):
            pass
        filename_list, article_list = os.listdir(load_path)[batch_start:batch_start + 100], []
        for filename in filename_list:
            with open(os.path.join(load_path, filename), 'r', encoding='UTF-8') as file:
                data = file.read()
            article_list.append(data)
        predict_source = vectorizer.transform(article_list)
        weight = transformer.transform(predict_source).toarray()

        total_result = []
        for sample_index in range(len(weight)):
            result = {'filename': filename_list[sample_index], 'words': []}
            for top_index in range(top_n):
                result['words'].append(
                    [reverse_vocabulary[numpy.argmax(weight[sample_index])], numpy.max(weight[sample_index])])
                weight[sample_index][numpy.argmax(weight[sample_index])] = -9999
            total_result.append(result)
        # print(total_result[0])
        json.dump(total_result, open(save_path + 'CNNDM_SalientWords_Train_%d_Top100.json' % batch_start, 'w'))
