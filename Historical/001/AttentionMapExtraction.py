import os
import torch
import tqdm
import json
import numpy
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer

save_path = 'D:/ProjectData/CNN_AttentionMap/'
batch_size = 16
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('D:/PythonProject/bert-base-uncased')
    mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)[0]
    model = BertModel.from_pretrained('D:/PythonProject/bert-base-uncased')
    model.eval()
    model.cuda()

    # if os.path.exists(os.path.join(save_path, filename.replace('story', 'json'))): continue
    # with open(os.path.join(save_path, filename.replace('story', 'json')), 'w') as file:
    #     pass
    with open("000c835555db62e319854d9f8912061cdca1893e.story", 'r', encoding='UTF-8')as file:
        raw_data = file.readlines()

    treat_article = ''
    for sample in raw_data:
        if sample[0] == '@': break
        treat_article += ' ' + sample.replace('\n', '').lower()

    treat_article = tokenizer.decode(tokenizer.encode(treat_article, max_length=512, truncation=True),
                                     skip_special_tokens=True)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([treat_article])
    word = vectorizer.get_feature_names()
    word = ['cardinals']
    # print(word)
    # exit()
    batch_word_tokens = tokenizer.batch_encode_plus([' ' + _ for _ in word], add_special_tokens=False)['input_ids']

    article_tokens = tokenizer.encode_plus(treat_article, max_length=512, truncation=True)['input_ids']
    masked_article_tokens = []
    for index in range(len(batch_word_tokens)):
        current_article_tokens = article_tokens.copy()
        appoint_keyword = batch_word_tokens[index]
        for indexX in range(len(article_tokens)):
            similar_flag = True
            for indexY in range(len(appoint_keyword)):
                if appoint_keyword[indexY] != article_tokens[indexX + indexY]:
                    similar_flag = False
                    break
            if similar_flag:
                for indexY in range(len(appoint_keyword)):
                    current_article_tokens[indexX + indexY] = mask_id
        masked_article_tokens.append(current_article_tokens)

    result = model(input_ids=torch.LongTensor(article_tokens).unsqueeze(0).cuda(), output_attentions=True)
    attention_map = result.attentions[0].mean(dim=1)
    for index in range(1, len(result.attentions)):
        attention_map += result.attentions[index].mean(dim=1)

    total_distance = []
    for batch_start in range(0, len(masked_article_tokens), batch_size):
        batch_data = masked_article_tokens[batch_start:batch_start + batch_size]
        batch_data = torch.LongTensor(batch_data).cuda()
        result = model(input_ids=batch_data, output_attentions=True)
        batch_attention_map = result.attentions[0].mean(dim=1)
        for index in range(1, len(result.attentions)):
            batch_attention_map += result.attentions[index].mean(dim=1)

        batch_distance = batch_attention_map - torch.cat([attention_map for _ in range(len(batch_data))], dim=0)
        batch_distance = batch_distance.sum(dim=-1).sum(dim=-1)
        total_distance.extend(batch_distance.detach().cpu().numpy())
    print(numpy.shape(batch_attention_map), numpy.shape(attention_map))
    batch_attention_map = batch_attention_map.squeeze().detach().cpu().numpy()
    attention_map = attention_map.squeeze().detach().cpu().numpy()

    import matplotlib.pylab as plt

    # print(numpy.shape(batch_attention_map), numpy.shape(attention_map))
    # data = (batch_attention_map - attention_map)
    # with open('result.csv', 'w') as file:
    #     for indexX in range(len(data)):
    #         for indexY in range(len(data[indexX])):
    #             file.write(str(data[indexX][indexY]) + ',')
    #         file.write('\n')

    # plt.imshow(numpy.log(numpy.abs(batch_attention_map - attention_map)))
    print(numpy.sum(numpy.abs(batch_attention_map - attention_map)))
    plt.imshow(numpy.log(attention_map))
    plt.show()
    # plt.imshow(attention_map)
    # plt.show()
