import os
import json
import tqdm
import numpy
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

if __name__ == '__main__':
    load_path = 'D:/ProjectData/CNNDM/Step0_TextTruncation'
    corpus = []
    for filename in tqdm.tqdm(os.listdir(load_path)):
        with open(os.path.join(load_path, filename), 'r', encoding='UTF-8') as file:
            part_data = file.read()
        corpus.append(part_data)

    vectorizer = CountVectorizer(analyzer='word')
    article_source = vectorizer.fit_transform(corpus)
    reverse_vocabulary = {}
    for key in vectorizer.vocabulary_:
        reverse_vocabulary[vectorizer.vocabulary_[key]] = key

    # 类调用
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(article_source)
    # weight = transformer.transform(predict_source).toarray()
    # print(numpy.shape(weight))
    pickle.dump(vectorizer, open('TFIDF_CountVectorizer.pkl', 'wb'))
    pickle.dump(transformer, open('TFIDF_TfidfTransformer.pkl', 'wb'))
