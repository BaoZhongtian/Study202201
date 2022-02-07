import os
import tqdm


def NCLS_Loader(sample_number=1000, use_part='train'):
    with open('C:/ProjectData/NCLS-Data/EN2ZHSUM/EN2ZHSUM_%s.txt' % use_part, 'r',
              encoding='UTF-8') as file:
        total_result, treat_sample = [], []
        treat_sentence = file.readline()

        while treat_sentence:
            treat_sample.append(treat_sentence.replace('\n', '').replace('\t', ''))
            if treat_sentence[0:6] == '</doc>':
                assert treat_sample[1] == '<Article>'
                article_sentence = treat_sample[2]
                for search_index in range(3, len(treat_sample)):
                    if treat_sample[search_index] == '</Article>': break
                    article_sentence += ' ' + treat_sample[search_index]

                summary, cross_lingual_summary = '', ''
                for index, sample in enumerate(treat_sample):
                    # print(index, sample)
                    if sample[0:7] == '<ZH-REF' and sample.find('-human-corrected>') == -1:
                        cross_lingual_summary += treat_sample[index + 1] + ' '
                    if sample[0:7] == '<EN-REF':
                        summary += treat_sample[index + 1] + ' '

                total_result.append(
                    {'Article': article_sentence, 'Summary': summary, 'CrossLingualSummary': cross_lingual_summary})
                treat_sample = []
                if len(total_result) >= sample_number: return total_result
                print('\rCurrent Load %d Samples' % len(total_result), end='')
            treat_sentence = file.readline()
    return total_result


if __name__ == '__main__':
    result = NCLS_Loader(sample_number=999999)
    print('\n', len(result))
    # print(result[2357])
    exit()
    for sample in result[0]:
        print(sample, result[0][sample])
