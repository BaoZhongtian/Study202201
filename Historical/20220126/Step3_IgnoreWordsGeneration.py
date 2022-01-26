import os
import json
import tqdm

if __name__ == '__main__':
    load_path = 'D:/ProjectData/CNNDM/Step2_KeywordsGeneration_Train'
    total_keywords_dictionary = {}
    for filename in tqdm.tqdm(os.listdir(load_path)):
        data = json.load(open(os.path.join(load_path, filename), 'r', encoding='UTF-8'))
        for indexX in range(len(data)):
            for indexY in range(len(data[indexX]['words'][0:20])):
                if data[indexX]['words'][indexY][0] in total_keywords_dictionary.keys():
                    total_keywords_dictionary[data[indexX]['words'][indexY][0]] += 1
                else:
                    total_keywords_dictionary[data[indexX]['words'][indexY][0]] = 1

    repeat_list = []
    for key in total_keywords_dictionary.keys():
        repeat_list.append([key, total_keywords_dictionary[key]])
    repeat_list = sorted(repeat_list, key=lambda x: x[-1], reverse=True)
    with open('IgnoreWords.txt', 'w', encoding='UTF-8') as file:
        for index in range(100): file.write(str(repeat_list[index][0]) + '\n')
