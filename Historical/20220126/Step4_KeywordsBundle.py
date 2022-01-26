import os
import json
import tqdm

selected_keywords_number = 20

if __name__ == '__main__':
    ignore_words = set()
    with open('IgnoreWords.txt', 'r', encoding='UTF-8') as file:
        data_raw = file.readlines()
    for sample in data_raw:
        ignore_words.add(sample.replace('\n', ''))

    keywords_path = 'D:/ProjectData/CNNDM/Step2_KeywordsGeneration_Train'
    total_keywords_dictionary = []
    for filename in tqdm.tqdm(os.listdir(keywords_path)):
        part_data = json.load(open(os.path.join(keywords_path, filename), 'r', encoding='UTF-8'))
        for treat_sample in part_data:
            treated_result = {'filename': treat_sample['filename'], 'words': []}
            search_index = -1
            while len(treated_result['words']) < selected_keywords_number:
                search_index += 1
                if treat_sample['words'][search_index][0].lower() in ignore_words: continue
                treated_result['words'].append(treat_sample['words'][search_index])
            total_keywords_dictionary.append(treat_sample)
    json.dump(total_keywords_dictionary, open('TFIDF_Keywords_Train.json', 'w', encoding='UTF-8'))
    print('Keywords Selected Completed')
