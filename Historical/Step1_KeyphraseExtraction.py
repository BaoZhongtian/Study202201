import os
import json
import tqdm
import numpy
from rouge_score import rouge_scorer
from rouge_score.tokenize import tokenize

top_n = 5

if __name__ == '__main__':
    load_path = 'D:/ProjectData/CNNDM/cnn/stories'
    # load_path = 'D:/ProjectData/CNNDM/dailymail_stories/dailymail/stories'
    save_path = 'D:/ProjectData/CNNDM_Treatment/Step1_KeyphraseExtraction'

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for filename in tqdm.tqdm(os.listdir(load_path)):
        with open(os.path.join(load_path, filename), 'r', encoding='UTF-8') as file:
            data = file.readlines()

        article, summary, summary_flag = [], '', False
        for sentence in data:
            sentence = sentence.replace('\n', '').lower()
            if sentence == '': continue
            if sentence[0] == '@':
                summary_flag = True
                continue
            if summary_flag:
                summary += sentence + ' '
            else:
                article.append(sentence)

        rouge_list = []
        for sentence in article:
            scores = scorer.score(target=summary, prediction=sentence)
            rouge_list.append(scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure)
        selected_sentences = []
        for index in range(top_n):
            selected_sentences.append(tokenize(article[numpy.argmax(rouge_list)], stemmer=False))
            rouge_list[numpy.argmax(rouge_list)] = 0

        selected_summary = tokenize(summary, stemmer=False)
        print(selected_summary)
        total_keyphrase = set()
        for treat_sentence in selected_sentences:
            for indexX in range(len(treat_sentence)):
                for indexY in range(len(selected_summary)):
                    if treat_sentence[indexX] != selected_summary[indexY]: continue
                    current_keyphrase = ''
                    for indexZ in range(min(len(selected_summary), len(treat_sentence) - indexX)):
                        if treat_sentence[indexX + indexZ] == selected_summary[indexY + indexZ]:
                            current_keyphrase += treat_sentence[indexX + indexZ] + ' '
                        else:
                            break
                    print(current_keyphrase)
        exit()
